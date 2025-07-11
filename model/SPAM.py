import torch
from torch import nn
from torch.nn import functional as f
import random

#DWConv 深度可分离卷积
class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        # x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.conv(x)  # (b c h w)
        # x = x.permute(0, 2, 3, 1)  # (b h w c)
        return x

class cWCT(torch.nn.Module):
    def __init__(self, eps=2e-5, use_double=False):
        super().__init__()
        self.eps = eps
        self.use_double = use_double

    def transfer(self, cont_feat, styl_feat, x_real, cmask=None, smask=None):
        if cmask is None or smask is None:
            return self._transfer(cont_feat, styl_feat, x_real)
        else:
            return self._transfer_seg(cont_feat, styl_feat, cmask, smask)

    def _transfer(self, cont_feat, styl_feat, x_real):
        """
        :param cont_feat: [B, N, cH, cW]
        :param styl_feat: [B, N, sH, sW]
        :return color_fea: [B, N, cH, cW]
        """
        B, N, cH, cW = cont_feat.shape
        cont_feat = cont_feat.reshape(B, N, -1)
        styl_feat = styl_feat.reshape(B, N, -1)
        x_real = x_real.reshape(B, N, -1)
        in_dtype = cont_feat.dtype
        if self.use_double:
            cont_feat = cont_feat.double()
            styl_feat = styl_feat.double()

        # whitening and coloring transforms
        whiten_fea = self.whitening(cont_feat, x_real)
        color_fea = self.coloring(whiten_fea, styl_feat)

        if self.use_double:
            color_fea = color_fea.to(in_dtype)

        return color_fea.reshape(B, N, cH, cW)

    def _transfer_seg(self, cont_feat, styl_feat, cmask, smask):
        """
        :param cont_feat: [B, N, cH, cW]
        :param styl_feat: [B, N, sH, sW]
        :param cmask: numpy [B, _, _]
        :param smask: numpy [B, _, _]
        :return color_fea: [B, N, cH, cW]
        """
        B, N, cH, cW = cont_feat.shape
        _, _, sH, sW = styl_feat.shape
        cont_feat = cont_feat.reshape(B, N, -1)
        styl_feat = styl_feat.reshape(B, N, -1)
        x_real = x_real.reshape(B, N, -1)

        in_dtype = cont_feat.dtype
        if self.use_double:
            cont_feat = cont_feat.double()
            styl_feat = styl_feat.double()

        for i in range(B):
            label_set, label_indicator = self.compute_label_info(cmask[i], smask[i])
            resized_content_segment = self.resize(cmask[i], cH, cW)
            resized_style_segment = self.resize(smask[i], sH, sW)

            single_content_feat = cont_feat[i]  # [N, cH*cW]
            single_style_feat = styl_feat[i]  # [N, sH*sW]
            target_feature = single_content_feat.clone()  # [N, cH*cW]

            for label in label_set:
                if not label_indicator[label]:
                    continue

                content_index = self.get_index(resized_content_segment, label).to(single_content_feat.device)
                style_index = self.get_index(resized_style_segment, label).to(single_style_feat.device)
                if content_index is None or style_index is None:
                    continue

                masked_content_feat = torch.index_select(single_content_feat, 1, content_index)
                masked_style_feat = torch.index_select(single_style_feat, 1, style_index)
                whiten_fea = self.whitening(masked_content_feat)
                _target_feature = self.coloring(whiten_fea, masked_style_feat)

                new_target_feature = torch.transpose(target_feature, 1, 0)
                new_target_feature.index_copy_(0, content_index,
                                               torch.transpose(_target_feature, 1, 0))
                target_feature = torch.transpose(new_target_feature, 1, 0)

            cont_feat[i] = target_feature
        color_fea = cont_feat

        if self.use_double:
            color_fea = color_fea.to(in_dtype)

        return color_fea.reshape(B, N, cH, cW)

    def cholesky_dec(self, conv, invert=False):
        cholesky = torch.linalg.cholesky if torch.__version__ >= '1.8.0' else torch.cholesky
        try:
            L = cholesky(conv)
        except RuntimeError:
            # print("Warning: Cholesky Decomposition fails")
            iden = torch.eye(conv.shape[-1]).to(conv.device)
            eps = self.eps
            while True:
                try:
                    conv = conv + iden * eps
                    L = cholesky(conv)
                    break
                except RuntimeError:
                    eps = eps + self.eps

        if invert:
            L = torch.inverse(L)

        return L.to(conv.dtype)

    def whitening(self, x, x_real=None):
        mean = torch.mean(x, -1)
        mean = mean.unsqueeze(-1).expand_as(x)
        x = x - mean

        conv = (x @ x.transpose(-1, -2)).div(x.shape[-1] - 1)
        inv_L = self.cholesky_dec(conv, invert=True)
        whiten_x = inv_L @ x

        return whiten_x

    def coloring(self, whiten_xc, xs):
        xs_mean = torch.mean(xs, -1)
        xs = xs - xs_mean.unsqueeze(-1).expand_as(xs)

        conv = (xs @ xs.transpose(-1, -2)).div(xs.shape[-1] - 1)
        Ls = self.cholesky_dec(conv, invert=False)

        coloring_cs = Ls @ whiten_xc
        coloring_cs = coloring_cs + xs_mean.unsqueeze(-1).expand_as(coloring_cs)

        return coloring_cs

    def compute_label_info(self, cont_seg, styl_seg):
        if cont_seg.size is False or styl_seg.size is False:
            return
        max_label = np.max(cont_seg) + 1
        self.label_set = np.unique(cont_seg)
        self.label_indicator = np.zeros(max_label)
        for l in self.label_set:
            # if l==0:
            #   continue
            is_valid = lambda a, b: a > 10 and b > 10 and a / b < 100 and b / a < 100
            o_cont_mask = np.where(cont_seg.reshape(cont_seg.shape[0] * cont_seg.shape[1]) == l)
            o_styl_mask = np.where(styl_seg.reshape(styl_seg.shape[0] * styl_seg.shape[1]) == l)
            self.label_indicator[l] = is_valid(o_cont_mask[0].size, o_styl_mask[0].size)
        return self.label_set, self.label_indicator

    def resize(self, img, H, W):
        size = (W, H)
        if len(img.shape) == 2:
            return np.array(Image.fromarray(img).resize(size, Image.NEAREST))
        else:
            return np.array(Image.fromarray(img, mode='RGB').resize(size, Image.NEAREST))

    def get_index(self, feat, label):
        mask = np.where(feat.reshape(feat.shape[0] * feat.shape[1]) == label)
        if mask[0].size <= 0:
            return None
        return torch.LongTensor(mask[0])

    def interpolation(self):
        # To do
        return

def split(x):
    n = int(x.size()[1] / 2)
    x1 = x[:, :n, :, :].contiguous()
    x2 = x[:, n:, :, :].contiguous()
    return x1, x2

def merge(x1, x2):
    return torch.cat((x1, x2), 1)

class residual_block(nn.Module):
    def __init__(self, channel, stride=1, mult=4, kernel=3, dropout=0.25):
        super().__init__()
        self.stride = stride

        pad = (kernel - 1) // 2
        if stride == 1:
            in_ch = channel
        else:
            in_ch = channel // 4

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_ch, channel // mult, kernel_size=kernel, stride=stride, padding=0, bias=True),
            nn.LeakyReLU(0.02, inplace=True),
            nn.ReflectionPad2d(pad),
            nn.Conv2d(channel // mult, channel // mult, kernel_size=kernel, padding=0, bias=True),
            nn.LeakyReLU(0.02, inplace=True),
            nn.ReflectionPad2d(pad),
            nn.Conv2d(channel // mult, channel, kernel_size=kernel, padding=0, bias=True)
        )
        with torch.no_grad():
            self.dropout = torch.nn.Dropout(dropout)
        self.init_layers()

    def init_layers(self):
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                # m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.conv(x2)
        if self.stride == 2:
            x1 = squeeze(x1)
            x2 = squeeze(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        x1, x2 = split(x)
        if self.stride == 2:
            x2 = unsqueeze(x2)
        Fx1 = self.conv(x1)
        if self.stride == 2:
            x1 = unsqueeze(x1)
        x1 = Fx1 + x2
        x = merge(x1, x2)
        return x
    # channels


class channel_reduction(nn.Module):
    def __init__(self, in_ch, out_ch, sp_steps=2, n_blocks=2, kernel=3):
        super().__init__()
        self.pad = out_ch * 4 ** sp_steps - in_ch
        #         self.inj_pad = injective_pad(self.pad)
        self.sp_steps = sp_steps
        self.n_blocks = n_blocks
        self.block_list = nn.ModuleList()
        for i in range(n_blocks):
            self.block_list.append(residual_block(in_ch, stride=1, mult=4, kernel=kernel))

    def forward(self, x):
        x = list(split(x))
        #         x[0] = self.inj_pad.forward(x[0])
        #         x[1] = self.inj_pad.forward(x[1])
        for block in self.block_list:
            x = block.forward(x)
        out = merge(x[0], x[1])
        return out


def val2list(x: list or tuple or any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def get_same_padding(kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


class ConvLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            stride=1,
            dilation=1,
            groups=1,
            use_bias=False,
            dropout=0,
            norm=False,
            act_func="relu",
    ):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.act:
            x = self.act(x)
        return x



class CR(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.channels = channel_reduction(in_ch=in_planes // 2, out_ch=16)
        # self.cwct = cWCT()

    def forward(self, content):
        cr = self.channels(content)
        return cr


class Coarse_render(nn.Module):
    def __init__(self):
        super().__init__()
        self.cwct = cWCT()

    def forward(self, content, style):
        cr = self.cwct.transfer(content, style, content)
        return cr

class MSA(nn.Module):
    def __init__(self, in_planes, max_sample=256 * 256):
        super().__init__()
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample
        self.dropout = nn.Dropout(0.0)
        self.out_conv1 = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.leap = DWConv2d(in_planes, 7, 1, 3)
        self.q1 = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.k1 = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.v1 = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.gelu = nn.GELU()

    def forward(self, style):
        AG_q = self.q1(style)
        s_k = self.k1(style)
        s_v = self.v1(style)
        head = 8
        b, c, h, w = AG_q.size()
        AG_q = AG_q.view(b, head, c // head, h, w)  # mutil head
        s_k = s_k.view(b, head, c // head, h, w)
        s_v = s_v.view(b, head, c // head, h, w)

        AG_q = AG_q.view(b, head, -1, w * h)  # C x HsWs

        s_k = s_k.view(b, head, -1, w * h).permute(0, 1, 3, 2)  # C x C
        AS = torch.einsum("bhci,bhik->bhck", AG_q, s_k)
        as1 = AS
        AS = self.sm(AS) * int(c // head) ** 0.5  # aesthetic attention map
        s_v = s_v.view(b, head, -1, w * h)
        astyle = torch.einsum("bhck,bhkj->bhcj", AS, s_v)
        DA = astyle.view(b, head, c // head, h, w).reshape(b, c, h, w)
        # DS Conv
        leak = self.gelu(self.leap(DA))
        leak += DA
        DA_local = self.out_conv1(leak)
        DA = DA_local + style
        return DA, as1


mlp = nn.ModuleList([nn.Linear(64, 64),
                     nn.ReLU(),
                     nn.Linear(64, 16),
                     nn.Linear(128, 128),
                     nn.ReLU(),
                     nn.Linear(128, 32),
                     nn.Linear(256, 256),
                     nn.ReLU(),
                     nn.Linear(256, 64),
                     nn.Linear(512, 512),
                     nn.ReLU(),
                     nn.Linear(512, 128)])


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """ avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3 """

        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class AdaIN(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y):
        content_mean, content_std = calc_mean_std(x)
        y_beta, y_gamma = calc_mean_std(y)  # sty_std *(c_features - c_mean) / c_std + sty_mean
        normalized_features = y_gamma * (x - content_mean) / content_std + y_beta
        return normalized_features


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.reshape(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(N, C, 1, 1)
    feat_mean = feat.reshape(N, C, -1).mean(dim=2).reshape(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat, mean, std


def mean_variance_normv2(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def block(x, patch_size=4, stride=4):
    b, c, h, w = x.shape
    r = int(patch_size ** 2)
    y = f.unfold(x, kernel_size=patch_size, stride=stride)
    y = y.permute(0, 2, 1)
    y = y.view(b, -1, c, r).permute(0, 2, 1, 3)
    # y = y.reshape(b,-1,r,c)      #b*c*l*r
    return y


def unblock(x, patch_size, stride, h):
    b, c, l, r = x.shape
    x = x.permute(0, 2, 1, 3)
    x = x.contiguous().view(b, l, -1).permute(0, 2, 1)
    y = f.fold(x, h, kernel_size=patch_size, stride=stride)
    # norm_map = f.fold(f.unfold(torch.ones(x.shape)))
    return y

# 编码器
class LiftPool(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(LiftPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # 获取输入维度
        batch_size, channels, height, width = x.size()

        # 计算输出维度
        out_height = height // self.stride
        out_width = width // self.stride

        # 将输入张量重塑为 (batch_size, channels, out_height, stride, out_width, stride)
        x_reshaped = x.view(batch_size, channels, out_height, self.stride, out_width, self.stride)

        # 沿着最后两个维度应用最大池化
        x_pooled, _ = x_reshaped.max(dim=3, keepdim=True)
        x_pooled, _ = x_pooled.max(dim=5, keepdim=True)

        # 重塑回原始维度
        x_pooled = x_pooled.view(batch_size, channels, out_height, out_width)

        return x_pooled
    # class fusion(nn.Module):


#     def __init__(self):
#         super(fusion, self).__init__()
#         self.downconv2 = nn.Conv2d(128, 256, 1, 1, 0)
#         self.downconv3 = nn.Conv2d(256, 256, 1, 1, 0)

#     def forward(self,sty_key3,sty_key2):
#         style_3 = self.downconv3(interpolate(sty_key3,scale_factor=0.5, mode='bilinear'))
#         style_2 = self.downconv2(interpolate(sty_key2, scale_factor=0.25, mode='bilinear'))
#         style_23 = torch.cat([style_2, style_3], dim=1)
#         return style_23

class fusion(nn.Module):
    def __init__(self):
        super(fusion, self).__init__()
        self.lift_poolv2 = LiftPool(kernel_size=4, stride=4)
        self.lift_poolv3 = LiftPool(kernel_size=2, stride=2)
        self.downconv2 = nn.Conv2d(128, 256, 1, 1, 0)
        self.downconv3 = nn.Conv2d(256, 256, 1, 1, 0)

    def forward(self, sty_key3, sty_key2):
        style_3 = self.downconv3(self.lift_poolv3(sty_key3))
        style_2 = self.downconv2(self.lift_poolv2(sty_key2))
        style_23 = torch.cat([style_2, style_3], dim=1)
        return style_23


def mean_normv2(feat, mean, std, a=0.5):
    mean1, std1 = calc_mean_std(feat)
    # normalized_feat = (feat) * std + mean
    normalized_feat = std * ((feat - mean1) / std1) + mean
    return normalized_feat


def mean_normv3(feat, mean, a=0.5):
    mean1, std1 = calc_mean_std(feat)
    # normalized_feat = (feat) * std + mean
    normalized_feat = (feat - mean1) + mean
    return normalized_feat


class SPAM(nn.Module):
    def __init__(self, in_planes, max_sample=256 * 256):
        super(SPAM, self).__init__()
        self.q = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.k = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.v = nn.Conv2d(in_planes, in_planes, (1, 1))

        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample
        self.fusion1 = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.CR = CR(in_planes)
        self.MSA = MSA(in_planes)
        self.cWCT_T = Coarse_render()

    def forward(self, content, style, threshold=0.5):
        region_l = 4
        stride_l = 4
        b, c, h, w = content.shape
        head = 8
        # full
        f_cr = self.CR(content)
        f_sa, logitss = self.MSA(style)
        f_cr = self.cWCT_T(f_cr, f_sa)
        norm_content = mean_variance_normv2(f_cr)
        norm_style = mean_variance_normv2(f_sa)
        Q1 = self.q(norm_content)
        K1 = self.k(norm_style)
        V1 = self.v(f_sa)
        # 4x4 block
        DA_q = block(Q1, region_l, stride_l)
        b, c, n, r = DA_q.shape
        DA_q = DA_q.view(b, head, c // head, n, r)
        # 卷积聚合
        # DA_k = block(K1, region_l, stride_l).view(b, head, c // head, n, r).permute(1, 0, 2, 3, 4)
        # DA_v = block(V1, region_l, stride_l).view(b, head, c // head, n, r).permute(1, 0, 2, 3, 4)
        # DA_k_all = torch.empty_like(DA_k).to(DA_k.device)
        # DA_v_all = torch.empty_like(DA_v).to(DA_v.device)
        # for i in range(head):
        #     DA_k_all[i] =  self.A2KV(DA_k[i])
        #     DA_v_all[i] =  self.A2KV(DA_v[i])
        # DA_k_all = DA_k_all.permute(1, 0, 2, 3, 4)
        # DA_v_all = DA_v_all.permute(1, 0, 2, 3, 4)
        DA_k_all = block(K1, region_l, stride_l).view(b, head, c // head, n, r)
        DA_v_all = block(V1, region_l, stride_l).view(b, head, c // head, n, r)
        DA_k_all = DA_k_all.expand_as(DA_q).reshape(b, head, c // head, -1)
        DA_v_all = DA_v_all.expand_as(DA_q).reshape(b, head, c // head, -1)
        DA_q = DA_q.reshape(b, head, c // head, -1)
        # Partition Attention
        logits = torch.einsum("bhci,bhki->bhck", DA_q, DA_k_all)
        logits1 = logits * (logits >= logitss)
        logits2 = logitss * (logits < logitss)
        logits_high = logits1 + logits2
        del logits1, logits2
        logits11 = logits * (logits <= logitss)
        logits22 = logitss * (logits > logitss)
        logits_low = logits11 + logits22
        del logits11, logits22
        mean_high, std_high = calc_mean_std(logits_high)
        mean_low, std_low = calc_mean_std(logits_low)
        mean_avg = (mean_high + mean_low) / 2
        std_avg = (std_high + std_low) / 2
        del mean_high, mean_low
        new_logits = mean_normv2(logits, mean_avg, std_avg)
        scores = self.sm(new_logits) * int(c // head) ** 0.5
        DA = torch.einsum("bhcc,bhcj->bhcj", scores, DA_v_all)
        DA_unblock = unblock(DA.contiguous().view(b, c, n, r), region_l, stride_l, h)
        Z_l = self.fusion1(DA_unblock)
        Z = Z_l + f_cr
        return Z


# Mutil-Chunk Attention
class MSAA(nn.Module):
    def __init__(self):
        super(MSAA, self).__init__()
        # self.MTAA_4=SANet(in_planes=512)
        # self.MTAA_3 = SANet(in_planes=512)
        self.MTAA_4 = SPAM(in_planes=512, max_sample=256 * 256)  # Ours
        self.MTAA_3 = SPAM(in_planes=512, max_sample=256 * 256)
        self.fusion = fusion()

        self.sim_alpha = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, content, style, sty_key3, sty_key2):
        cs_feature_4 = self.MTAA_4(content, style)
        style_l = self.fusion(sty_key3, sty_key2)
        cs_feature_3 = self.MTAA_3(content, style_l)
        # cs_feature = self.gamma * cs_feature_3 + (1-self.gamma) * cs_feature_4
        cs_feature = (1 - 0.5 * self.sim_alpha) * cs_feature_3 + 0.5 * self.sim_alpha * cs_feature_4
        return cs_feature, cs_feature_4, cs_feature_3, style_l

