import torch
from torch import nn
import time as t
from PIL import Image
from tqdm import tqdm
from torch import optim
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from torch.utils import data
from torch.nn.functional import interpolate
import kornia.augmentation as K
import cv2
import os
from data_processing import *
from model.SPAM import *
from model.VGGNet import *
from model.GuideDiscriminator import *

# main Model
class CSPANet(object):
    def __init__(self, args):
        super(CSPANet, self).__init__()
        # 定义配置
        self.scaler = torch.cuda.amp.GradScaler()
        self.device = args.device
        self.result_dir = args.result_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.vgg19_dir = args.vgg_dir
        self.dataset = args.dataset
        self.content_dir = args.content_dir
        self.style_dir = args.style_dir
        self.video_dir = args.video_dir
        self.test_dir = args.test_dir
        self.init_train = args.init_train
        self.isTrain = args.isTrain
        self.isTest = args.isTest
        self.style_path = args.style_path
        self.cpu_count = args.cpu_count
        self.n_res = args.n_res
        self.dimn = args.dimn
        self.PONO = args.PONO
        self.max_iter = args.max_iter
        # 定义模型参数
        self.input_c = args.input_c
        self.hw = args.hw
        self.b1 = args.b1
        self.b2 = args.b2
        self.g_lr = args.g_lr
        self.d_lr = args.d_lr
        self.decay_lr = args.decay_lr
        self.latent_dim = args.latent_dim
        self.s = args.s
        self.batch_size = args.batch_size
        self.save_pred = args.save_pred
        # 模型权重参数w
        self.weight_content = args.weight_content
        self.weight_style = args.weight_style
        # 定义模型
        self.vgg = VGG.to(self.device)
        self.vgg.load_state_dict(torch.load(self.vgg19_dir))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:44])
        self.vggcoder = vggcoder(self.vgg).to(self.device)
        self.decoder = DECODER.to(self.device)
        self.decoder = Decoder(self.decoder).to(self.device)
        self.GD = GuideDiscriminator(in_channels=3).to(self.device)
        self.SPAM = MSAA().to(self.device)
        self.cwct = cWCT()
        self.SPAM.apply(init_weights)
        self.GD.apply(init_weights)
        #         self.decoder.apply(init_weights)
        # 定义优化器
        self.optim_SPAM = optim.Adam(self.SPAM.parameters(), lr=self.g_lr, betas=(self.b1, self.b2))
        self.optim_Decoder = optim.Adam(self.decoder.parameters(), lr=self.g_lr, betas=(self.b1, self.b2))
        self.optim_GD = optim.Adam(self.GD.parameters(), lr=self.d_lr, betas=(self.b1, self.b2))
        # 定义损失
        self.l1_loss = nn.L1Loss()
        self.huber = nn.SmoothL1Loss()
        self.mse_loss = nn.MSELoss()
        self.adain = AdaIN()
        self.tv_loss = VariationLoss(1)
        self.bce_loss = nn.BCEWithLogitsLoss()
        #         self.ssim = SSIM()
        self.cos = torch.nn.CosineSimilarity(eps=1e-6)
        #         self.ssim.to(self.device)
        self._rgb_to_yuv_kernel = torch.tensor([
            [0.299, -0.14714119, 0.61497538],
            [0.587, -0.28886916, -0.51496512],
            [0.114, 0.43601035, -0.10001026]
        ]).float().to(self.device)
        # 打印配置
        print('---------- Networks initialized -------------')
        print("##### Information #####")
        print("# device : ", self.device)
        print(f"线程:{int(self.cpu_count)}")
        print("# dataset : ", self.dataset)
        print(f'# DECODER :{print_network(self.decoder) / 1e6}M')
        print(f'# DisGA :{print_network(self.SPAM) / 1e6}M')
        print(f'# GD:{print_network(self.GD) / 1e6}M')
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.max_iter)
        print("# training image size [H, W] : ", self.hw)
        print("# contents,style: ", self.weight_content, self.weight_style)
        print("# g_lr,d_lr: ", self.g_lr, self.d_lr)
        print('-----------------------------------------------')

    # 读取数据集
    def load_data(self, epoch_test=False, high=False, video=False):
        self.mode = epoch_test
        self.high = high
        self.video = video
        # 增强操作
        #         transforms.Resize([256,256]),
        train_trans = [transforms.Resize(286),
                       transforms.CenterCrop(256),
                       transforms.RandomHorizontalFlip(0.5),
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])]
        test_trans = [transforms.Resize([256, 256]),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])]
        test_high_trans = [transforms.Resize([512, 512]),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])]
        test_video_trans = [transforms.Resize([1024, 1024]),
                            transforms.ToTensor()]

        transform_list = transforms.Compose([
            transforms.Resize(size=(286, 286)),
            transforms.RandomCrop(256),
            transforms.ToTensor()
        ])

        content_tf = transform_list
        style_tf = transform_list

        content_dataset = FlatFolderDataset(self.content_dir, content_tf)
        style_dataset = FlatFolderDataset(self.style_dir, style_tf)

        content_iter = iter(data.DataLoader(content_dataset, batch_size=self.batch_size,
                                            sampler=InfiniteSamplerWrapper(content_dataset),
                                            num_workers=12))
        style_iter = iter(
            data.DataLoader(style_dataset, batch_size=self.batch_size, sampler=InfiniteSamplerWrapper(style_dataset),
                            num_workers=12))
        #         if self.mode and self.isTrain:
        #             data_loader = data.DataLoader(Test_ImagePools(root=self.data_dir, trans=test_trans),
        #                                           batch_size=4,
        #                                           pin_memory=True,
        #                                           drop_last=True
        #                                           , num_workers=12)
        #         if self.isTrain and not self.mode:
        #             data_loader = data.DataLoader(ImagePools(root=self.data_dir, trans=train_trans),
        #                                           batch_size=self.batch_size, pin_memory=True,shuffle=True,
        #                                           drop_last=True
        #                                           , num_workers=12)

        #         if self.isTest:
        #             data_loader = data.DataLoader(Test_Eval(root=self.data_dir, trans=test_trans), batch_size=1, num_workers=2,
        #                                           pin_memory=True)
        #         if self.high:
        #             data_loader = data.DataLoader(Test_high(trans=test_high_trans), batch_size=1,
        #                                           num_workers=0, pin_memory=True)
        #         if self.video:
        #             data_loader = data.DataLoader(Test_Video_TestDataset(root=self.data_dir, trans=test_video_trans),
        #                                           batch_size=1, num_workers=0, pin_memory=True)
        return content_iter, style_iter

        # 加载灰度patch

    def texture_exact(self, real, fake):
        # 进行灰度颜色改变
        # real_gry = self.color2trans.process(real)
        # fake_gry = self.color2trans.process(fake)
        real_gry = color_shift(real)
        fake_gry = color_shift(fake)
        return real_gry, fake_gry

    def resize(self, x):
        trans = transforms.Resize([512, 512])
        out = trans(x)
        return out

    # conten loss
    def con_loss(self, fake, real):
        # _, c, w, h = fake.shape
        #         real, fake= self.texture_exact(real,fake)
        out_con = self.l1_loss(fake, real)
        return out_con

    def style_loss(self, fake, real):
        _, c, w, h = fake.shape
        out_con = self.mse_loss(gram(fake), gram(real))
        return out_con

    def cycle_loss(self, fake, real):
        _, c, w, h = fake.shape
        fake_u, fake_std = calc_mean_std(fake)
        #         print(fake_u.shape,fake_std.shape)
        real_u, real_std = calc_mean_std(real)
        out_con = self.mse_loss(fake_u, real_u) + self.mse_loss(fake_std, real_std)
        return out_con

    def cycle_lossv2(self, fake, real):
        _, c, w, h = fake.shape
        fake_u, fake_std = calc_mean_std(fake)
        real_u, real_std = calc_mean_std(real)
        fake_mean = torch.mean(fake, dim=1)
        real_mean = torch.mean(real, dim=1)

        out_con = self.mse_loss(fake_u, real_u) + self.mse_loss(fake_std, real_std) + self.mse_loss(fake_mean,
                                                                                                    real_mean)
        return out_con

    def mse_loss(self, input, target):
        return torch.mean((input - target) ** 2)

    def CWCT(self, content, style):
        return self.cwct.transfer(content, style, content)
        # return self.adain(content, style)

    def extract_image_patches(self, x, kernel, stride=1):
        b, c, h, w = x.shape

        # Extract patches
        patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        patches = patches.contiguous().view(b, c, -1, kernel, kernel)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()

        return patches.view(b, -1, c, kernel, kernel)

    def content_loss(self, x, y):
        x_gray, y_gray = self.texture_exact(x, y)
        x_ = self.gf(x_gray, x_gray, r=5, eps=2e-1)
        y_ = self.gf(y_gray, y_gray, r=5, eps=2e-1)
        return self.mse_loss(x_, y_)

    def adjust_learning_rate(self, optimizer, iteration_count):
        lr = self.g_lr / (1.0 + 5e-5 * iteration_count)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # 训练
    def train(self):
        content_iter, style_iter = self.load_data()
        count = len(content_iter)
        start_t = t.time()
        Res_skip = {}
        real, fake = 1, 0
        self.decoder.train(), self.SPAM.train()
        print('==========================start train=====================================')
        iter = 0
        for i in tqdm(range(0, self.max_iter)):
            #                     if (epoch+1) > 1:
            self.adjust_learning_rate(self.optim_SPAM, i)
            self.adjust_learning_rate(self.optim_Decoder, i)
            self.adjust_learning_rate(self.optim_GD, i)
            x = next(content_iter).to(self.device)
            y = next(style_iter).to(self.device)

            # zero grident
            self.optim_SPAM.zero_grad(set_to_none=True)
            self.optim_Decoder.zero_grad(set_to_none=True)
            self.optim_GD.zero_grad(set_to_none=True)

            # coarse process
            c_code, skipc = self.vggcoder(x)  # output contents tensor of every layers
            s_code, skips = self.vggcoder(y)
            Res_skip['cwct_1'] = self.CWCT(skipc['relu_3'], skips['relu_3'])
            Res_skip['cwct_2'] = self.CWCT(skipc['relu_2'], skips['relu_2'])
            # put style tensor of every layers
            cs, _, _, _ = self.SPAM(c_code[-2], s_code[-2], skips['relu_3'], skips['relu_2'])
            fake_image = self.decoder(cs, Res_skip)
            # adv d loss
            adv_d_loss = (self.GD.compute_loss(fake_image, fake) + self.GD.compute_loss(y, real)) * 2
            adv_d_loss.backward()
            self.optim_GD.step()

            self.optim_SPAM.zero_grad(set_to_none=True)
            self.optim_Decoder.zero_grad(set_to_none=True)
            self.optim_GD.zero_grad(set_to_none=True)

            # contents loss
            c_code, skipc = self.vggcoder(x)  # output contents tensor of every layers
            s_code, skips = self.vggcoder(y)
            Res_skip['cwct_1'] = self.CWCT(skipc['relu_3'], skips['relu_3'])
            Res_skip['cwct_2'] = self.CWCT(skipc['relu_2'], skips['relu_2'])
            # put style tensor of every layers
            cs, cs4, cs3, _ = self.SPAM(c_code[-2], s_code[-2], skips['relu_3'], skips['relu_2'])
            fake_image = self.decoder(cs, Res_skip)
            # adv g loss
            adv_g_loss = self.GD.compute_loss(fake_image, real) * 2
            # contents loss
            fake_code, skipcs = self.vggcoder(fake_image)
            _, _, _, sty3 = self.SPAM(fake_code[-2], fake_code[-2], skipcs['relu_3'], skipcs['relu_2'])
            # attention loss
            attention_loss4 = self.mse_loss(cs4, fake_code[-2]) * 1.5
            attention_loss3 = self.mse_loss(cs3, sty3) * 1.5
            attention_loss = attention_loss3 + attention_loss4

            con_loss = self.mse_loss(fake_code[-2], c_code[-2]) * 1.5
            # style loss
            style_loss = (self.cycle_loss(fake_code[0], s_code[0]) + self.cycle_loss(fake_code[1], s_code[1]) +

                          self.cycle_loss(fake_code[2], s_code[2]) + self.cycle_loss(fake_code[3], s_code[3])) * 5
            total_loss = adv_g_loss + style_loss + con_loss + attention_loss
            total_loss.backward()
            self.optim_SPAM.step()
            self.optim_Decoder.step()
            end_epoch_t = t.time()
            if (i + 1) % 100 == 0 or i == 0:
                print(
                    f"epoch:[iter:[{i + 1}/{count}],loss_g:{adv_g_loss},loss_d:{adv_d_loss},attention:{attention_loss},content_loss:{con_loss},loss_style:{style_loss},G_lr:{self.optim_Decoder.param_groups[0]['lr']},time:{time_change(end_epoch_t - start_t)}")
            if (i + 1) % self.save_pred == 0:
                with torch.no_grad():
                    #                         self.save_img(epoch)
                    self.save_modelv2()
                    print(f'{iter}')

    # 保存模型
    def save_model(self):
        params = {}
        params["DisA"] = self.SPAM.state_dict()
        params["Decoder"] = self.decoder.state_dict()
        torch.save(params, os.path.join(self.result_dir, self.dataset, 'checkpoint',
                                        f'checkpoint_{self.dataset}.pth'))
        print("保存模型成功！")

    # 保存模型
    def save_modelv2(self):
        params = {}
        params["DisA"] = self.SPAM.state_dict()
        params["GD"] = self.GD.state_dict()
        params["Decoder"] = self.decoder.state_dict()
        torch.save(params, os.path.join(self.result_dir, self.dataset, 'checkpoint',
                                        f'checkpoint_{self.dataset}.pth'))
        print("保存模型成功！")

    # 加载模型
    def load_model(self):
        params = torch.load(self.test_dir)
        self.SPAM.load_state_dict(params['DisA'])
        self.decoder.load_state_dict(params['Decoder'])
        self.GD.load_state_dict(params['GD'])
        print("加载模型成功！")

    # 高分辨图像测试
    def high_test(self):
        Res_skip = {}
        self.load_model()
        test_sample_num = 1
        self.decoder.eval(), self.SPAM.eval()
        data_loader = self.load_data(high=True)
        count = 0
        for i, (x, _) in tqdm(enumerate(data_loader)):
            x = x[i].to(self.device)
            for j, (_, y) in tqdm(enumerate(data_loader)):
                y = y[j].to(self.device)
                with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
                    c_code, skipc = self.vggcoder(x)  # output contents tensor of every layers
                    s_code, skips = self.vggcoder(y)
                    Res_skip['cwct_1'] = self.CWCT(skipc['relu_3'], skips['relu_3'])
                    Res_skip['cwct_2'] = self.CWCT(skipc['relu_2'], skips['relu_2'])
                    cs = self.SPAM(c_code[-2], s_code[-2], skips['relu_3'], skips['relu_2'])
                    fake_image = self.decoder(cs, Res_skip)
                print(prof)
                prof.export_chrome_trace('profiles')
                fake_img = denormalzation(fake_image, self.device)
                save_image(fake_img, os.path.join(self.result_dir, self.dataset, 'img', f"{i}_{j}.jpg"))
        print("高分辨率测试集测试图像生成成功！")

    def loadImg(self, imgPath):
        img = Image.open(imgPath).convert('RGB')
        transform = transforms.Compose([transforms.Resize([512, 512]),
                                        transforms.ToTensor()
                                        ])
        return transform(img)

    def video_test(self):
        Res_skip = {}
        result_frames = []
        contents = []
        self.load_model()
        test_sample_num = 1
        self.decoder.eval(), self.adain.eval()
        data_loader = self.load_data(video=True)
        print(len(data_loader))
        styleV = self.loadImg(self.style_path).unsqueeze(0)
        styleV = styleV.to(self.device)
        style = styleV.squeeze(0).cpu().numpy()
        style_code, skips = self.vggcoder(styleV)  #
        for i, x in tqdm(enumerate(data_loader)):
            x1 = x.to(self.device)
            con_key, sty_key = self.zzz(x1, styleV)
            content_code, skipc = self.vggcoder(x1)  # output contents tensor of every layers
            Res_skip['cwct_1'] = self.CWCT(skipc['relu_3'], skips['relu_3'])
            Res_skip['cwct_2'] = self.CWCT(skipc['relu_2'], skips['relu_2'])
            intermediate_code = self.adain(content_code[-2], style_code[-2], con_key, sty_key)
            fake_img = self.decoder(intermediate_code, Res_skip)
            # x, fake_img = utils.denormalzation(x1, self.device), utils.denormalzation(fake_img, self.device)
            #             save_image(x,
            #                        os.path.join(self.result_dir, self.dataset, 'test_c', f'{i}.png'))
            contents.append(x1.squeeze(0).float().cpu().numpy())
            result_frames.append(fake_img.squeeze(0).cpu().detach().numpy())
        out_name = os.path.join(self.result_dir, self.dataset, 'video')
        self.makeVideo(contents, style, result_frames, outf=out_name)
        print("高分辨率测试集测试图像生成成功！")

    def makeVideo(self, content, style, props, outf):
        print('Stack transferred frames back to video...')
        layers, height, width = content[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video = cv2.VideoWriter(os.path.join(outf, 'transfer.avi'), fourcc, 10.0, (width, height))
        ori_video = cv2.VideoWriter(os.path.join(outf, 'contents.avi'), fourcc, 10.0, (width, height))
        for j in range(len(content)):
            prop, cont = utils.numpy2cv2(content[j], style, props[j], width, height)
            cv2.imwrite('prop.jpg', prop)
            cv2.imwrite('contents.jpg', cont)
            # TODO: this is ugly, fix this
            imgj = cv2.imread('prop.jpg')
            imgc = cv2.imread('contents.jpg')

            video.write(imgj)
            ori_video.write(imgc)
            # RGB or BRG, yuks
        video.release()
        ori_video.release()
        os.remove('prop.jpg')
        os.remove('contents.jpg')
        print('Transferred video saved at %s.' % outf)