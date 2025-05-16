

import numpy as np

from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop


import torch

import cv2

from torchvision import transforms as trans
import os
# import libnvjpeg
# import pickle
#todo 按照原始顺序存储


img_root_dir = '/media/HDD1/wangtao/lunwen8/original_protecteds/shouyetu/'
save_path = '/media/HDD1/wangtao/lunwen8/original_protecteds/shouyetu_224/'

device = torch.device('cuda:0')
# device = torch.device('cpu')

opt = TestOptions().parse()
start_epoch, epoch_iter = 1, 0
crop_size = opt.crop_size
# threshold = 1.54
test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
crop_size=224
# decoder = libnvjpeg.py_NVJpegDecoder()
if crop_size == 512:
    opt.which_epoch = 550000
    opt.name = '512'
    mode = 'ffhq'
else:
    mode = 'None'
embed_map = {}
app = Face_detect_crop(name='antelope', root='./insightface_func/models')
app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640), mode=mode)

for root, dirs, files in os.walk(img_root_dir):
    for name in files:
        if name.endswith('jpg') or name.endswith('png'):

           try:
                p = os.path.join(root, name)
                img = cv2.imread(p)
                # if img.shape[0]>256 and img.shape[1]
                face, _ = app.get(img, crop_size)
                # new_path = name[0:-4]
                new_path = name
                a=os.path.join(save_path, new_path)
                b=np.array(face[0])
                # print(a)
                print(b.shape)
                cv2.imwrite(a, b)

                # face.save(a)
                    # embed_map[new_path] = embed.detach().cpu()
           except Exception as e:
               print(name+'----------------------------------------')
               continue



