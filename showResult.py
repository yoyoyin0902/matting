import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image

# model = torch.jit.load('model/pytorch_resnet50.pth').cuda().eval() 
"""
RuntimeError: Attempting to deserialize object on a CUDA device but 
torch.cuda.is_available() is False. If you are running on a CPU-only machine, 
please use torch.load with map_location=torch.device('cpu') to map your 
storages to the CPU.
"""

model = torch.load('model/pytorch_resnet50.pth',map_location ='cpu')

src = Image.open('images/img/12.png')
bgr = Image.open('images/bgr/12.png')

src = to_tensor(src)
bgr = to_tensor(bgr)

if src.size(1) <= 2048 and src.size(2) <= 2048:
  model.backbone_scale = 1/4
  model.refine_sample_pixels = 80_000
else:
  model.backbone_scale = 1/8
  model.refine_sample_pixels = 320_000

pha, fgr = model(src, bgr)[:2]

com = pha * fgr + (1 - pha) * torch.tensor([120/255, 255/255, 155/255], device='cuda').view(1, 3, 1, 1)

to_pil_image(com[0].cpu())

to_pil_image(pha[0].cpu()).save('pha.png')
to_pil_image(fgr[0].cpu()).save('fgr.png')
to_pil_image(com[0].cpu()).save('com.png')
