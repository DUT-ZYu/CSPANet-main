import os
import cv2
import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
import torchvision.utils as vutils
# from torchscan.crawler import crawl_module
# from fvcore.nn import FlopCountAnalysis
import time as t

def load_segment(image, image_size=None):
	if image_size is not None:
		transform = transforms.Resize(image_size, interpolation=Image.NEAREST)
		image = transform(image)
	w, h = image.size
	transform = transforms.CenterCrop((h // 16 * 16, w // 16 * 16))
	image = transform(image)
	if len(np.asarray(image).shape) == 3:
		image = change_seg(image)
	return np.asarray(image)

class GuidedFilter(nn.Module):
    def box_filter(self, x: torch.Tensor, r):
        ch = x.shape[1]
        k = 2 * r + 1
        weight = 1 / ((k) ** 2)  # 1/9
        # [c,1,3,3] * 1/9
        box_kernel = torch.ones((ch, 1, k, k), dtype=torch.float32, device=x.device).fill_(weight)
        # same padding
        return torch.nn.functional.conv2d(x, box_kernel, padding=r, groups=ch)

    def forward(self, x: torch.Tensor, y: torch.Tensor, r, eps=1e-2):
        b, c, h, w = x.shape
        device = x.device

        N = self.box_filter(torch.ones((1, 1, h, w), dtype=x.dtype, device=device), r)

        mean_x = self.box_filter(x, r) / N
        mean_y = self.box_filter(y, r) / N
        cov_xy = self.box_filter(x * y, r) / N - mean_x * mean_y
        var_x = self.box_filter(x * x, r) / N - mean_x * mean_x

        A = cov_xy / (var_x + eps)
        b = mean_y - A * mean_x

        mean_A = self.box_filter(A, r) / N
        mean_b = self.box_filter(b, r) / N

        output = mean_A * x + mean_b
        return output

def attn_visualization_all(Ic, Is, Fc, Fs, Attn, slide_size=4):
    if Ic.size(3) != Is.size(3):
        Is = torch.nn.functional.interpolate(Is, size=(Ic.size(2), Ic.size(3)))
    B, C, S = Attn.size()
    B, c, c_w, c_h = Ic.size()
    B, c, s_w, s_h = Is.size()
    _, _, f_w, f_h = Fs.size()
    attn = torch.zeros(1, Fs.size(2) * Fs.size(3)).cuda()
    mask = torch.zeros(1, Fc.size(2) * Fc.size(3)).cuda()
    for index in range(0, Fc.size(2) * Fc.size(3), slide_size):
        for idx in range(slide_size):
            start_idx = index + Fc.size(3) * idx
            start_idx_ref = index + Fs.size(3) * idx
            mask[0][start_idx:start_idx + slide_size] = 1
            attn += torch.sum(Attn[0][start_idx:start_idx + slide_size], dim=0)
    masked = torch.nn.functional.interpolate(mask.view(1, 1, Fc.size(2), Fc.size(3)), size=(c_w, c_h), mode='nearest')
    attn = torch.nn.functional.interpolate(attn.view(1, 1, Fs.size(2), Fs.size(3)), size=(s_w, s_h), mode='nearest')
    # imsave_no_norm(Is, attn.repeat(1, 3, 1, 1), 'style.png')
    return attn.repeat(1, 3, 1, 1), Ic * (masked + 0.2), Is * (attn * (1 / attn.max()) + 0.2), Is * (attn * 0.05 + 0.2)

# 参数权重初始化
def init_weights(m, init_type='normal', gain=0.02):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, gain)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            torch.nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        elif init_type == 'kaiming':
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight.data, gain=gain)
        elif init_type == 'none':  # uses pytorch's default init method
            m.reset_parameters()
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


# 梯度
def requires_grad(model, flag=True):
    if model is None:
        return
    for p in model.parameters():
        p.requires_grad = flag


# 时间转化
def time_change(time):
    new_time = t.localtime(time)
    new_time = t.strftime("%Hh%Mm%Ss", new_time)
    return new_time


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


# 归一化
def tensor2numpy(x):
    return x.cpu().detach().numpy().transpose(1, 2, 0)


def denormalzation(tensor, device):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
    tensor = torch.clamp(tensor * std + mean, 0., 1.)
    return tensor

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

class VariationLoss(nn.Module):
    def __init__(self, k_size: int) -> None:
        super().__init__()
        self.k_size = k_size

    def forward(self, image: torch.Tensor):
        b, c, h, w = image.shape
        tv_h = torch.mean((image[:, :, self.k_size:, :] - image[:, :, : -self.k_size, :]) ** 2)
        tv_w = torch.mean((image[:, :, :, self.k_size:] - image[:, :, :, : -self.k_size]) ** 2)
        tv_loss = (tv_h + tv_w) / (3 * h * w)
        return tv_loss

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params

def feature_normalize(feature_in, eps=1e-10):
    feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + eps
    feature_in_norm = torch.div(feature_in, feature_in_norm)
    return feature_in_norm

def gram(input):
    b, c, w, h = input.size()
    x = input.view(b * c, w * h)
    G = torch.mm(x, x.T)
#     print(G.shape)
    return G.div(b * c * w * h)

def gram_matrix(y):
	(b, ch, h, w) = y.size()
	features = y.view(b, ch, w * h)
	features_t = features.transpose(1, 2)
	gram = features.bmm(features_t) / (ch * h * w)
	return gram

# 颜色RGB->Gray(DCT-NET)
def color_shift(image, mode='uniform'):
    device = image.device
    b1, g1, r1 = torch.split(image, 1, dim=1)
    if mode == 'normal':
        b_weight = torch.normal(mean=0.114, std=0.1, size=[1]).to(device)
        g_weight = torch.normal(mean=0.587, std=0.1, size=[1]).to(device)
        r_weight = torch.normal(mean=0.299, std=0.1, size=[1]).to(device)
    elif mode == 'uniform':
        b_weight = torch.FloatTensor(1).uniform_(0.014, 0.214).to(device)
        g_weight = torch.FloatTensor(1).uniform_(0.487, 0.687).to(device)
        r_weight = torch.FloatTensor(1).uniform_(0.199, 0.399).to(device)
    output1 = (b_weight * b1 + g_weight * g1 + r_weight * r1) / (b_weight + g_weight + r_weight)
    return output1

# RGB2LAB
def rgb2xyz(img):
    """
    RGB from 0 to 255
    :param img:
    :return:
    """
    r, g, b = torch.split(img, 1, dim=1)

    r = torch.where(r > 0.04045, torch.pow((r + 0.055) / 1.055, 2.4), r / 12.92)
    g = torch.where(g > 0.04045, torch.pow((g + 0.055) / 1.055, 2.4), g / 12.92)
    b = torch.where(b > 0.04045, torch.pow((b + 0.055) / 1.055, 2.4), b / 12.92)

    r = r * 100
    g = g * 100
    b = b * 100

    x = r * 0.412453 + g * 0.357580 + b * 0.180423
    y = r * 0.212671 + g * 0.715160 + b * 0.072169
    z = r * 0.019334 + g * 0.119193 + b * 0.950227
    return torch.cat([x, y, z], dim=1)

def xyz2lab(xyz):
    x, y, z = torch.split(xyz, 1, dim=1)
    ref_x, ref_y, ref_z = 95.047, 100.000, 108.883
    # ref_x, ref_y, ref_z = 0.95047, 1., 1.08883
    x = x / ref_x
    y = y / ref_y
    z = z / ref_z

    x = torch.where(x > 0.008856, torch.pow(x, 1 / 3), (7.787 * x) + (16 / 116.))
    y = torch.where(y > 0.008856, torch.pow(y, 1 / 3), (7.787 * y) + (16 / 116.))
    z = torch.where(z > 0.008856, torch.pow(z, 1 / 3), (7.787 * z) + (16 / 116.))

    l = (116. * y) - 16.
    a = 500. * (x - y)
    b = 200. * (y - z)
    return torch.cat([a, b], dim=1)

def rgb_to_yuv(image, x):
    image = (image + 1.0) / 2.0
    yuv_img = torch.tensordot(
        image,
        x,
        dims=([image.ndim - 3], [0]))
    return yuv_img

def adjust_learning_rate(optimizer, lr, lr_decay, iteration_count):
    lr -= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def print_params(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')

def __write_images(image_outputs, display_image_num, file_name, normalize=False):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]  # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=normalize)
    vutils.save_image(image_grid, file_name, nrow=1)

def write_2images(image_outputs, display_image_num, image_directory, postfix, normalize=False):
    __write_images(image_outputs, display_image_num, '%s/gen_a2b_%s.jpg' % (image_directory, postfix), normalize)

def write_one_row_html(html_file, iterations, img_filename, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (iterations, img_filename.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img1 src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
    return

def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" contents="60">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    write_one_row_html(html_file, iterations, '%s/gen_a2b_train_current.jpg' % (image_directory), all_size)
    for j in range(iterations, image_save_iterations - 1, -image_save_iterations):
        if j % image_save_iterations == 0:
            write_one_row_html(html_file, j, '%s/gen_a2b_train_%08d.jpg' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()


def save_video(video, save_path, type='photo'):
    video = denorm(video)
    '''
    vid_lst=[]
    for i in range(0, num_samples*num_samples, num_samples):
        temp_vid = list(video[i:i+num_samples])
        temp_vid = torch.cat(temp_vid, dim=-1)
        vid_lst.append(temp_vid)

    save_videos = torch.cat(vid_lst, dim=2)
    '''

    save_videos = video.data.cpu().numpy().transpose(0, 2, 3, 1)
    outputdata = save_videos * 255
    # outputdata = ((save_videos+1)/2) * 255
    outputdata = outputdata.astype(np.uint8)
    dir_path = save_path
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if type == 'photo':
        gif_file_path = os.path.join(dir_path, 'Photo_StylizedVideo.gif')
    elif type == 'art':
        gif_file_path = os.path.join(dir_path, 'Art_StylizedVideo.gif')
    else:
        gif_file_path = os.path.join(dir_path, 'content_StylizedVideo.gif')
    imageio.mimsave(gif_file_path, outputdata, fps=25)


def numpy2cv2(cont, style, prop, width, height):
    cont = cont.transpose((1, 2, 0))
    cont = cont[..., ::-1]
    cont = cont * 255
    cont = cv2.resize(cont, (width, height))
    # cv2.resize(iimg,(width,height))
    style = style.transpose((1, 2, 0))
    style = style[..., ::-1]
    style = style * 255
    style = cv2.resize(style, (width, height))

    prop = prop.transpose((1, 2, 0))
    prop = prop[..., ::-1]
    prop = prop * 255
    prop = cv2.resize(prop, (width, height))
    return prop, cont

def img_resize(img, max_size, down_scale=None):
    w, h = img.size
    if max(w, h) > max_size:
        w = int(1.0 * img.size[0] / max(img.size) * max_size)
        h = int(1.0 * img.size[1] / max(img.size) * max_size)
        img = img.resize((w, h), Image.BICUBIC)
    if down_scale is not None:
        w = w // down_scale * down_scale
        h = h // down_scale * down_scale
        img = img.resize((w, h), Image.BICUBIC)
    return img

def parse_shapes(input):
    if isinstance(input, list) or isinstance(input,tuple):
        out_shapes = [item.shape[1:] for item in input]
    else:
        out_shapes = input.shape[1:]
    return out_shapes

