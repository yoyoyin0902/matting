# -*- coding: utf-8 -*-
"""BackgroundMattingV2-ImageMatting.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cTxFq1YuoJ5QPqaTcnskwlHDolnjBkB9

# BackgroundMattingV2 Image Matting Example

This notebook demonstrates how to use our model for image matting. For more information about our project, please visit [our github repo](https://github.com/PeterL1n/BackgroundMattingV2).

## Prepare images

You can upload your own images as `src.png` and `bgr.png` using the left panel.

Or download our example images using the commands below. More example images are available at [Google Drive](https://drive.google.com/drive/folders/16H6Vz3294J-DEzauw06j4IUARRqYGgRD?usp=sharing).
"""

# !pip install gdown -q

# !gdown https://drive.google.com/uc?id=1g_n7kzDppAA2BNwUfit0eVKXLHRVBOw- -O src.png -q
# !gdown https://drive.google.com/uc?id=1ywAx_GKGswFQjt3rWBLjdLtzjrjiOWpG -O bgr.png -q

"""## Matting

Download our model file. All of our model files can be found at [Google Drive](https://github.com/PeterL1n/BackgroundMattingV2#model--weights). Below we download the `torchscript_resnet50_fp32.pth` weights. The file contains the model architecture and is ready for production use.
"""

import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import gdown

# url = 'https://drive.google.com/u/0/uc?id=/1fMl7qepWqWvROlWvwLyr9TFGaAUBIYtW=download'
# https://drive.google.com/file/d/13mFNHX_vCcMrNHJgbWiyDD1XadceuBOw/view?usp=sharing 
# https://drive.google.com/file/d/1pqZKBvkh1a-wO-RMPN8krp03OvqIMjAT/view?usp=sharing

# output = 'model1.pth'
# gdown.download(url, output)
# print("1111")


"""# 新增區段"""
# model = BiGRU()	
# model.load_state_dict(torch.load('model.pth’))





model = torch.load('torchscript_mobilenetv2_fp32.pth').cuda().eval() 


src = Image.open('/home/user/all.png')
bgr = Image.open('/home/user/back.png')

src = to_tensor(src).cuda().unsqueeze(0)
bgr = to_tensor(bgr).cuda().unsqueeze(0)

if src.size(2) <= 2048 and src.size(3) <= 2048:
  model.backbone_scale = 1/4
  model.refine_sample_pixels = 80000 #800_00 = 80000
else:
  model.backbone_scale = 1/8
  model.refine_sample_pixels = 320000

pha, fgr = model(src, bgr)[:2]

com = pha * fgr + (1 - pha) * torch.tensor([120/255, 100/255, 125/255], device='cuda').view(1, 3, 1, 1)

to_pil_image(pha[0].cpu())

to_pil_image(com[0].cpu())


to_pil_image(pha[0].cpu()).save('pha.png')
to_pil_image(fgr[0].cpu()).save('fgr.png')
to_pil_image(com[0].cpu()).save('com.png')
