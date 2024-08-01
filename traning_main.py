
# encoding: utf-8
import argparse
from data import VGGFace
import torch.utils.data as data

import torch
import torchvision.utils as vutils
# todo
import torch.nn.functional as F
import os
from AIDPro import AIDPro
os.environ['CUDA_VISIBLE_DEVICES'] ='0'


class Progressbar():
    def __init__(self):
        self.p = None
    def __call__(self, iterable):
        from tqdm import tqdm
        self.p = tqdm(iterable)
        return self.p
    def say(self, **kwargs):
        if self.p is not None:
            self.p.set_postfix(**kwargs)



def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--FFM_pretrain', type=str,default='pretrained_models/90000_net_G.pth',help='load the pretrained model from the specified location')
    parser.add_argument("--Arc_path", type=str, default='models/arcface_checkpoint.tar', help="run ONNX model via TRT")
    parser.add_argument('--len_watermark', type=int, default=30)
    parser.add_argument('--data_path', default='/media/HDD1/wangtao/datatset/vggface/vggface2_crop_arcfacealign_224/',
                        type=str)
    parser.add_argument('--lambda_wa', type=float, default=10)
    parser.add_argument('--lambda_rec', type=float, default=5)
    parser.add_argument('--lambda_id', type=float, default=5)
    parser.add_argument('--epochs', dest='epochs', type=int, default=2, help='# of epochs')
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

train_dataset = VGGFace(args.data_path)

train_dataloader = data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    shuffle=True, drop_last=True

)

print('Training images:', len(train_dataset))

Mymodel = AIDPro(args)

progressbar = Progressbar()
imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
def mynorm(image):
    image_temp = image * imagenet_std
    return image_temp + imagenet_mean


it = 20000
it_per_epoch = len(train_dataset) // args.batch_size

for epoch in range(0,args.epochs):
    lr = args.lr_base / (10 ** (epoch // 100))
    Mymodel.set_lr(lr)

    for img_a in progressbar(train_dataloader):
        Mymodel.train()
        img_a = img_a.cuda() if args.gpu else img_a
        #
        errG = Mymodel.trainG(img_a)
        progressbar.say(epoch=epoch, iter=it + 1, sum_loss=errG['sum_loss'], id_loss=errG['id_loss'],
                        rec_loss=errG['rec_loss'], wa_loss=errG['wa_loss'])
        it += 1

        # todo 输出水印准确率
        if it % 1000 == 0:
            print(errG)
            Mymodel.eval()
            with torch.no_grad():
                sum_bit_acc = 0
                count_bit_acc = 0
                for imgs_a in train_dataloader:
                    if count_bit_acc * len(imgs_a) > 2000:
                        break
                    # 保护人脸
                    imgs_a = imgs_a.cuda() if args.gpu else imgs_a
                    imgs_fake, original_id, watermarks = Mymodel.generate_protected(imgs_a)
                    ## 水印准确度计算
                    fakes_id = Mymodel.netArc(F.interpolate(imgs_fake, (112, 112), mode='bicubic'))
                    fakes_id = F.normalize(fakes_id, p=2, dim=1)
                    extracted_watermarkings = Mymodel.WD(fakes_id)
                    extracted_watermarkings = (torch.sigmoid(extracted_watermarkings) > 0.5)
                    watermarks = watermarks > 0.5

                    sum_bit_acc = sum_bit_acc + (extracted_watermarkings == watermarks).float().mean()
                    # print((extracted_watermarking == watermark).float().mean())
                    # 保存图像
                    if count_bit_acc==0:
                        samples = [mynorm(imgs_a)]


                        # 保存图像

                        samples.append(mynorm(imgs_fake))
                        samples.append(mynorm(Mymodel.netG(imgs_a, original_id)))
                        samples = torch.cat(samples, dim=3)
                        filename = 'test_images/LPIPS_L1(0.2)_all_loss_' + '_id_' + str(args.lambda_id) + '_wa_' + str(
                            args.lambda_wa) + '_rec_' + str(args.lambda_rec) + '_step_' + str(
                            it) + '.jpg'
                        vutils.save_image(samples, filename, nrow=1)



                    count_bit_acc = count_bit_acc + 1
                print(sum_bit_acc / count_bit_acc)


        # 保存模型
        # 保存模型
        if it % 10000 == 0:
            Mymodel.save('premodels/LPIPS_L1(0.2)_all_loss'  + '_id_' + str(args.lambda_id) + '_rec_'+str(args.lambda_rec)+'_wa_'  +  str(args.lambda_wa) + '_step_' + str(
                it) + '.pt')













