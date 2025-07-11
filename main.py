import argparse
import os
from train import CSPANet
# from train_AdaIN import DFAGAN


def parse_args():
    desc='pytorch of CSPANet'
    parser=argparse.ArgumentParser(desc)
    parser.add_argument('--device',type=str,default='cuda',choices=['cuda','cpu'])
    parser.add_argument('--input_c', type=int, default=1)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--cpu_count', type=int, default=1)
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for ADAM')#3.789080525625532e-05
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for ADAM')
    parser.add_argument('--decay_lr', type=float, default=0.00002, help='learning rate for ADAM')
    parser.add_argument('--hw', type=int, default=256)
    parser.add_argument('--result_dir',type=str,default='results')
    parser.add_argument('--video_dir',type=str,default='/kaggle/input/videoss/videos/contents/mountain_2')
    parser.add_argument('--style_path',type=str,default='test_images/style1/16.jpg')
    parser.add_argument('--checkpoint_dir', type=str, default='Checkpoint/checkpoint_cartoon60最后.pth',
                        help='Name of checkpoint directory')
    parser.add_argument('--vgg_dir',type=str,default="/kaggle/input/itstyler-weight/vgg_normalised.pth")
    parser.add_argument('--dataset', type=str, default='Mystyle')
    parser.add_argument('--content_dir',type=str,default='/kaggle/input/coco-wikiart-nst-dataset-512-100000/content')
    parser.add_argument('--style_dir',type=str,default='/kaggle/input/coco-wikiart-nst-dataset-512-100000/style')
    parser.add_argument('--test_dir',type=str,default="/kaggle/input/mytrans3-8-4w/checkpoint_Mystyle.pth")
    parser.add_argument('--init_train', type=bool, default=False)
    parser.add_argument('--isTrain',type=bool,default=True)
    parser.add_argument('--retrain', type=bool, default=False)
    parser.add_argument('--isTest', type=bool, default=False)
    parser.add_argument('--isHigh', type=bool, default=True)
    parser.add_argument('--isVideo', type=bool, default=False)
    parser.add_argument('--b1',type=int,default=0.5)
    parser.add_argument('--b2', type=int, default=0.999)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--latent_dim2', type=int, default=512)
    parser.add_argument('--n_res', type=int, default=2)
    parser.add_argument('--dimn', type=int, default=64)
    parser.add_argument('--s', type=int, default=48)
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--save_pred',type=int,default=1000)
    parser.add_argument('--PONO', action='store_true', help='use positional normalization ')
    parser.add_argument('--weight_content',type=float,default=1)#big
    parser.add_argument('--weight_cty', type=float, default=3)  # big
    parser.add_argument('--weight_tv', type=float, default=1)
    parser.add_argument('--weight_style', type=float, default=5)
    parser.add_argument('--weight_g', type=float, default=5)
    parser.add_argument('--weight_d', type=float, default=5)
    return check_args(parser.parse_args(args=[]))
# 创建文件目录
def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def check_args(args):
    check_folder(os.path.join(args.result_dir, args.dataset, 'checkpoint'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'sty'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'con'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'video'))
    return args

def main():
   args = parse_args()
   gan = CSPANet(args)
   if args.isTrain:
       if args.retrain:
            gan.load_model()
       print(f"training on {args.device}")
       gan.train()
       print("train haved finished")
   if args.isTest:
       gan.FID_test()
       print("test haved finished")
   if args.isHigh:
       gan.high_test()
       print("test high haved finished")
   if args.isVideo:
       gan.video_test()
       print("test video haved finished")

if __name__=="__main__":
    main()
#测试精度