
# encoding: utf-8
import argparse
from data_test import CelebA
import torch.utils.data as data
import torch
import torchvision.utils as vutils
# todo
import torch.nn.functional as F
import os
from AIDPro import AIDPro
os.environ['CUDA_VISIBLE_DEVICES'] ='0'


def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--FFM_pretrain', type=str,
                        default='/media/HDD1/wangtao/lunwen6/SimSwap-main/checkpoints/simswap/90000_net_G.pth',
                        help='load the pretrained model from the specified location')

    parser.add_argument("--Arc_path", type=str, default='models/arcface_checkpoint.tar', help="run ONNX model via TRT")
    parser.add_argument('--len_watermark', type=int, default=30)

    parser.add_argument('--data_path', default='original_224/',type=str)

    parser.add_argument('--save_path', default='protected_224/',
                      type=str)

    parser.add_argument('--lambda_wa', type=float, default=100.0)
    parser.add_argument('--lambda_id', type=float, default=1)
    parser.add_argument('--lambda_rec', type=float, default=1)

    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--epochs', dest='epochs', type=int, default=1, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)  # todo
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.999)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)

    parser.add_argument('--n_samples', dest='n_samples', type=int, default=32, help='# of sample images')

    parser.add_argument('--gpu', dest='gpu', action='store_true', default=True)

    return parser.parse_args(args)

# seed_torch()
args = parse()
print(args)

args.lr_base = args.lr
args.betas = (args.beta1, args.beta2)


train_dataset = CelebA(args.data_path)

train_dataloader = data.DataLoader(
    train_dataset, batch_size=1,
    shuffle=False, drop_last=True
)

print('Training images:', len(train_dataset))

CanFG = AIDPro(args)
# todo 预训练模型
# CanFG.load('premodels/no_LPIPS.pt')
CanFG.load('pretrained_models/LPIPS_L1(0.2)_all_loss_id_5_rec_5_wa_10_step_60000.pt')
CanFG.eval()

imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
def mynorm(image):
    image_temp = image * imagenet_std
    return image_temp + imagenet_mean

def mynorm_2(image):
    image_temp = image / imagenet_std
    return image_temp - imagenet_mean

def mynorm_(image):
    image_temp = image - imagenet_mean
    return image_temp / imagenet_std

it = 1
it_per_epoch = len(train_dataset) // args.batch_size

CanFG.eval()
with torch.no_grad():

    sum_bit_acc = 0
    count_bit_acc = 0
    watermarks_sum=[]
    for original_imgs,name in train_dataloader:
        # 保护人脸
        original_imgs = original_imgs.cuda() if args.gpu else original_imgs

        imgs_fake, _, watermarks = CanFG.generate_protected(original_imgs)
        samples = [mynorm(imgs_fake)]
        names = name[0]
        # names = new_path = '%04d.png'%it
        # filename1='/media/HDD1/wangtao/lunwen6/watermarking/protected_faces/original_CelebA/'+names
        filename2 = args.save_path + names
        # filename2 = '/media/HDD1/wangtao/lunwen6/watermarking/protected_faces/protected/' + names
        # vutils.save_image(mynorm(original_imgs), filename1, nrow=1)
        vutils.save_image(mynorm(imgs_fake), filename2, nrow=1)
        watermarks_sum.append((watermarks[0].int().tolist()))

        # for i in range(20):
        #     imgs_fake, _, watermarks = CanFG.generate_protected(original_imgs)
        #     samples.append(mynorm(imgs_fake))
        #
        # samples = torch.cat(samples, dim=3)
        # vutils.save_image(samples, filename2, nrow=1)



        it = it + 1

    # torch.save(torch.tensor(watermarks_sum), '/media/HDD1/wangtao/lunwen6/watermarking/protected_faces/watermarks_no_LPIPS.pt')
    torch.save(torch.tensor(watermarks_sum), 'watermarks.pt')










# 测试水印准确度
# with torch.no_grad():
#     sum_bit_acc = 0
#     count_bit_acc = 0
#     for protected_imgs,original_imgs,watermarks in train_dataloader:
#         protected_imgs = protected_imgs.cuda()
#         original_imgs= original_imgs.cuda()
#         watermarks = watermarks.cuda()
#
#
#         noise_image=protected_imgs
#         #保存图像
#         samples = [mynorm(original_imgs)]
#         samples.append(mynorm(protected_imgs))
#         # if sum_bit_acc==0:
#         #     noise_image = F.interpolate(noise_image, (224, 224), mode='bicubic')
#         #     samples.append(mynorm(noise_image))
#         #     samples = torch.cat(samples, dim=3)
#         #     filename = 'test_images/robust_test_jpg.jpg'
#         #     vutils.save_image(samples, filename, nrow=1)
#
#         ## 水印准确度计算
#         fakes_id = CanFG.netArc(F.interpolate(noise_image, (112, 112), mode='bicubic'))
#         fakes_id = F.normalize(fakes_id, p=2, dim=1)
#         extracted_watermarkings = CanFG.WD(fakes_id)
#         extracted_watermarkings = (torch.sigmoid(extracted_watermarkings) > 0.5)
#         watermarks = watermarks > 0.5
#         count_bit_acc = count_bit_acc + 1
#         sum_bit_acc = sum_bit_acc + (extracted_watermarkings == watermarks).float().mean()
#         # print((extracted_watermarking == watermark).float().mean())
#     print(sum_bit_acc / count_bit_acc)




