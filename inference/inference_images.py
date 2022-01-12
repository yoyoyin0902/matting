"""
Inference images: Extract matting on images.

Example:

    python inference_images.py \
        --model-type mattingrefine \
        --model-backbone mobilenetv2 \
        --model-backbone-scale 0.25 \
        --model-refine-mode sampling \
        --model-refine-sample-pixels 80000 \
        --model-checkpoint "PATH_TO_CHECKPOINT" \
        --images-src "PATH_TO_IMAGES_SRC_DIR" \
        --images-bgr "PATH_TO_IMAGES_BGR_DIR" \
        --output-dir "PATH_TO_OUTPUT_DIR" \
        --output-type com fgr pha

"""

import argparse
import torch
import os
import shutil

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from threading import Thread
from tqdm import tqdm

from dataset import ImagesDataset, ZipDataset
from dataset import augmentation as A
from model import MattingBase, MattingRefine
from inference_utils import HomographicAlignment

class PairApply:
    def __init__(self, transforms):
        self.transforms = transforms
  
    def __call__(self, *x):
        return [self.transforms(xi) for xi in x]

  
class PairCompose(T.Compose):
    def __call__(self, *x):
        for transform in self.transforms:
            x = transform(*x)
        return x


# --------------- Arguments ---------------


parser = argparse.ArgumentParser(description='Inference images')

parser.add_argument('--model-type', type=str, required=False, choices=['mattingbase', 'mattingrefine'],default='mattingrefine')
parser.add_argument('--model-backbone', type=str, required=False, choices=['resnet101', 'resnet50', 'mobilenetv2'],default='mobilenetv2')
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-checkpoint', type=str, required=False,default='/home/user/matting/model_pth/pytorch_mobilenetv2.pth')
parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80000)
parser.add_argument('--model-refine-threshold', type=float, default=0.7)
parser.add_argument('--model-refine-kernel-size', type=int, default=3)

#parser.add_argument('--images-src', type=str, required=False,default="/home/user/matting/all.png")
#parser.add_argument('--images-bgr', type=str, required=False,default="/home/user/matting/back.png")

parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--num-workers', type=int, default=0, 
    help='number of worker threads used in DataLoader. Note that Windows need to use single thread (0).')
parser.add_argument('--preprocess-alignment', action='store_true')

parser.add_argument('--output-dir', type=str, required=False,default='/home/user/matting/img_output')
parser.add_argument('--output-types', type=str, required=True, nargs='+', 
    choices=['com', 'pha', 'fgr', 'err', 'ref'])
parser.add_argument('-y', action='store_true')




def test(image_path: str, bgr_path: str):
    parser.add_argument('--images-src', type=str, required=False, default=image_path)
    parser.add_argument('--images-bgr', type=str, required=False, default=bgr_path)
    args = parser.parse_args()

    assert 'err' not in args.output_types or args.model_type in ['mattingbase', 'mattingrefine'], \
    'Only mattingbase and mattingrefine support err output'
    assert 'ref' not in args.output_types or args.model_type in ['mattingrefine'], \
    'Only mattingrefine support ref output'

    # --------------- Main ---------------
    device = torch.device(args.device)

    # Load model
    if args.model_type == 'mattingbase':
        model = MattingBase(args.model_backbone)
    if args.model_type == 'mattingrefine':
        model = MattingRefine(
        args.model_backbone,
        args.model_backbone_scale,
        args.model_refine_mode,
        args.model_refine_sample_pixels,
        args.model_refine_threshold,
        args.model_refine_kernel_size)

    model = model.to(device).eval()
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=False)

    # Load images
    dataset = ZipDataset([
        ImagesDataset(args.images_src),
        ImagesDataset(args.images_bgr),
    ], assert_equal_length=True, transforms=A.PairCompose([
        HomographicAlignment() if args.preprocess_alignment else PairApply(nn.Identity()),
        PairApply(T.ToTensor())
    ]))
    dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)
    print(dataloader)

    # Create output directory
    # if os.path.exists(args.output_dir):
    #     if args.y or input(f'Directory {args.output_dir} already exists. Override? [Y/N]: ').lower() == 'y':
    #         shutil.rmtree(args.output_dir)
    #     else:
    #         exit()

    for output_type in args.output_types:
        if os.path.exists(os.path.join(args.output_dir, output_type)) is False:
            os.makedirs(os.path.join(args.output_dir, output_type))
 
    # Worker function
    def writer(img, path):
        img = to_pil_image(img[0].cpu())
        print(path)
        img.save(path)

    
    with torch.no_grad(): 
        for i, (src, bgr) in enumerate(tqdm(dataloader)):
            src = src.to(device, non_blocking=True)
            bgr = bgr.to(device, non_blocking=True)

            if args.model_type == 'mattingbase':
                pha, fgr, err, _ = model(src, bgr)
            elif args.model_type == 'mattingrefine':
                pha, fgr, _, _, err, ref = model(src, bgr)

            pathname = dataset.datasets[0].filenames[i]
            pathname = os.path.relpath(pathname, args.images_src)
            pathname = os.path.splitext(pathname)[0]
            
            if 'com' in args.output_types:
                com = torch.cat([fgr * pha.ne(0), pha], dim=1)
                Thread(target=writer, args=(com, os.path.join(args.output_dir, 'com', pathname + '.png'))).start()

            if 'pha' in args.output_types:
                Thread(target=writer, args=(pha, os.path.join(args.output_dir, 'pha', pathname + '.jpg'))).start()

            if 'fgr' in args.output_types:
                Thread(target=writer, args=(fgr, os.path.join(args.output_dir, 'fgr', pathname + '.jpg'))).start()

            if 'err' in args.output_types:
                err = F.interpolate(err, src.shape[2:], mode='bilinear', align_corners=False)
                Thread(target=writer, args=(err, os.path.join(args.output_dir, 'err', pathname + '.jpg'))).start()
            
            if 'ref' in args.output_types:
                ref = F.interpolate(ref, src.shape[2:], mode='nearest')
                Thread(target=writer, args=(ref, os.path.join(args.output_dir, 'ref', pathname + '.jpg'))).start()

# Conversion loop
if __name__ == '__main__':
    test("/home/user/matting/all.png", "/home/user/matting/back.png")