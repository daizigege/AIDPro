
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import lpips

import torch.nn as nn
from fs_networks_fix import Generator_Adain_Upsample


class Watermark_insert(nn.Module):
    def __init__(self, input_size, hidden_size=512, output_size=512):
        super(Watermark_insert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层的全连接层
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # Leaky ReLU激活函数
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层的全连接层

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        return x


class Watermark_decoder(nn.Module):
    def __init__(self, output_size, input_size=512, hidden_size=512):
        super(Watermark_decoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层的全连接层
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # Leaky ReLU激活函数
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层的全连接层

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        return x


class AIDPro(nn.Module):
    def __init__(self, args):
        super(AIDPro, self).__init__()
        self.len_watermark=args.len_watermark
        self.lambda_id=args.lambda_id
        self.lambda_wa = args.lambda_wa
        self.lambda_rec=args.lambda_rec
        # Generator network
        self.netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9).cuda()
        self.netG.load_state_dict(torch.load(args.FFM_pretrain))
        self.netG.eval()

        self.lpips_loss=lpips.LPIPS(net='alex').cuda()

        self.netArc = torch.load(args.Arc_path, map_location=torch.device("cpu")).cuda()
        self.netArc.eval()

        # 水印嵌入网络
        self.WI=Watermark_insert(args.len_watermark+512).cuda()
        self.WI.train()

        # 水印提取网络
        self.WD=Watermark_decoder(args.len_watermark).cuda()
        self.WD.train()

        self.optim_WI = optim.Adam(self.WI.parameters(), lr=args.lr, betas=args.betas)
        self.optim_WD = optim.Adam(self.WD.parameters(), lr=args.lr, betas=args.betas)


    def set_lr(self, lr):
        for g in self.optim_WI.param_groups:
            g['lr'] = lr
        for g in self.optim_WD.param_groups:
            g['lr'] = lr


    def generate_protected(self,img_a):
        with torch.no_grad():
            emb_img = self.netArc(F.interpolate(img_a, (112, 112), mode='bicubic'))
            original_id = F.normalize(emb_img, p=2, dim=1)
        watermark = torch.Tensor(np.random.choice([0, 1], (img_a.shape[0], self.len_watermark))).cuda()
        # print(watermark)
        concatenated_matrix = torch.cat((original_id, watermark), dim=1)
        concatenated_id = self.WI(concatenated_matrix)
        concatenated_id = F.normalize(concatenated_id, p=2, dim=1)

        ## 生成保护图像
        img_fake = self.netG(img_a, concatenated_id)

        return img_fake,original_id,watermark



    def trainG(self, img_a):

        img_fake,original_id,watermark=self.generate_protected(img_a)

        ## 保护人脸的提取身份
        fake_id=self.netArc(F.interpolate(img_fake, (112, 112), mode='bicubic'))
        fake_id = F.normalize(fake_id, p=2, dim=1)

        ## 保护人脸的提取水印
        extracted_watermarking=self.WD(fake_id)

        # 身份对抗损失 1-torch.cosine_similarity(mean_identities,emb_fake).mean()

        distance=torch.cosine_similarity(original_id, fake_id)

        loss_id=torch.maximum(distance,torch.tensor(0)).mean()

        loss_rec=0.2*torch.nn.functional.l1_loss(img_a,img_fake)+0.8*self.lpips_loss(img_a,img_fake).mean()

        # 水印提取损失
        loss_watermark=F.binary_cross_entropy_with_logits(extracted_watermarking,watermark)

        sum_loss = self.lambda_id*loss_id + self.lambda_rec*loss_rec+self.lambda_wa*loss_watermark#self.lambda_rec * gr_loss+self.lambda_em*EM_loss+self.lambda_lp*lpips_loss

        self.optim_WI.zero_grad()

        self.optim_WD.zero_grad()

        sum_loss.backward()
        self.optim_WI.step()
        self.optim_WD.step()
        errG = {
            'sum_loss': sum_loss.item(), 'id_loss': loss_id.item(),'rec_loss': loss_rec.item(),'wa_loss': loss_watermark.item()
        }
        return errG


    def train(self):
        self.WD.train()
        self.WI.train()

    def eval(self):
        self.WD.eval()
        self.WI.eval()

    def save(self, path):
        states = {
            'WD': self.WD.state_dict(),
            'WI': self.WI.state_dict(),
            'optim_WD': self.optim_WD.state_dict(),
            'optim_WI': self.optim_WI.state_dict(),
        }
        torch.save(states, path)

    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        # if 'G' in states:
        self.WD.load_state_dict(states['WD'])
        self.WI.load_state_dict(states['WI'])

        # if 'EM' in states:
        # try:
        #     # 可能会引发错误的代码
        #     self.WI.load_state_dict(states['WI'])
        # except Exception as e:
        #     # 如果发生 ZeroDivisionError 异常，这里的代码会执行
        #     print("虚拟身份提取器的模型参数不对")
        # if 'optim_G' in states:
        self.optim_WD.load_state_dict(states['optim_WD'])
        # if 'optim_D' in states:
        self.optim_WI.load_state_dict(states['optim_WI'])




