import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
from PIL import Image
import json
from datetime import datetime

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
from piq import DISTS

from nn_arch.RRDBNet_arch import RRDBNet
from dataloader import DF2KDataset


def validate():
    parser = argparse.ArgumentParser(description='ESRGAN Testing')
    parser.add_argument('--mode', type=str, default='orig', 
                        help='Mode: orig, rgb2, yuv')
    parser.add_argument('--scale', type=int, default=2, 
                        help='Scale parametr (2 or 4)')
    parser.add_argument('--save_num', type=int, default=10, 
                        help='Number of visual results to save')
    parser.add_argument('--output_dir', type=str, default='./validation_results')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.mode == 'rgb2':
        nf_base, gc_base = 32, 16
        print(f"Mode 'rgb2': number of channels (nf, gc) is halved (nf={nf_base}, gc={gc_base}).")
    elif args.mode == 'orig':
        nf_base, gc_base = 64, 32
        print(f"Mode 'orig': standart channels (nf={nf_base}, gc={gc_base}).")

    model = RRDBNet(in_nc=3, out_nc=3, nf=nf_base, nb=23, gc=gc_base).to(device)
    model.load_state_dict(torch.load(f'./checkpoints/esrgan_{args.mode}.pth', map_location=device))
    model.eval()
    print(f"Weights are downloaded: {args.weights}")

    val_dataset = DF2KDataset(split='val', scale=args.scale)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"Found {len(val_loader)} images in validation set.")

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = lpips.LPIPS(net='vgg').to(device)
    lpips_metric.eval()
    for p in lpips_metric.parameters():
        p.requires_grad = False
        
    dists_metric = DISTS().to(device)
    dists_metric.eval()
    for p in dists_metric.parameters():
        p.requires_grad = False

    total_psnr, total_ssim, total_lpips, total_dists = 0.0, 0.0, 0.0, 0.0
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Цикл валидации
    with torch.no_grad():
        for idx, (lr_imgs, hr_imgs) in enumerate(tqdm(val_loader, desc="Validation")):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            
            sr_imgs = torch.clamp(model(lr_imgs), 0, 1)

            total_psnr += psnr_metric(sr_imgs, hr_imgs).item()
            total_ssim += ssim_metric(sr_imgs, hr_imgs).item()
            total_dists += dists_metric(sr_imgs, hr_imgs).item()

            total_lpips += lpips_metric(sr_imgs * 2 - 1, hr_imgs * 2 - 1).item()

            if idx < args.save_num:
                lr_resized = torch.nn.functional.interpolate(lr_imgs, size=hr_imgs.shape[2:], mode='nearest')
                
                # апскейл соседа | Результат ESRGAN | Оригинал (HR)
                grid = make_grid(torch.cat([lr_resized, sr_imgs, hr_imgs], dim=0), nrow=3, padding=4)
                save_path = os.path.join(args.output_dir, f"result_sample_{idx:03d}.png")
                save_image(grid, save_path)

    num_samples = len(val_loader)
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_lpips = total_lpips / num_samples
    avg_dists = total_dists / num_samples

    print("\n")
    print(f"RESULTS (Scale: X{args.scale})")
    print(f"Avg PSNR:  {avg_psnr:.2f} dB")
    print(f"Avg SSIM:  {avg_ssim:.4f}")
    print(f"Avg LPIPS: {avg_lpips:.4f}")
    print(f"Avg DISTS: {avg_dists:.4f}")

    res = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "psnr": avg_psnr,
           "ssim": avg_ssim,
           "lpips": avg_lpips,
           "dists": avg_dists}

    res_path = os.path.join(args.output_dir, "metrics.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)

    print(f"Metrics are saved in: {res_path}")


def inference():

    parser = argparse.ArgumentParser(description='ESRGAN Inference')
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to LR image')
    parser.add_argument('--output', type=str, default='./results/output.png', 
                        help='Path to save upscaled image')
    parser.add_argument('--mode', type=str, default='orig', 
                        help='Mode: orig, rgb2, yuv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.mode == 'rgb2':
        nf_base, gc_base = 32, 16
        print(f"Mode 'rgb2': number of channels (nf и gc) is halved (nf={nf_base}, gc={gc_base}).")
    elif args.mode == 'orig':
        nf_base, gc_base = 64, 32
        print(f"Mode 'orig': standart channels (nf={nf_base}, gc={gc_base}).")

    model = RRDBNet(in_nc=3, out_nc=3, nf=nf_base, nb=23, gc=gc_base).to(device)
    model.load_state_dict(torch.load(f'./checkpoints/esrgan_{args.mode}.pth', map_location=device))
    model.eval()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image {args.input} is not found.")

    img = Image.open(args.input).convert('RGB')
    orig_w, orig_h = img.size
    print(f"Original shape: {orig_w}x{orig_h}")

    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device) # [1, 3, H, W]

    print("Start Upscaler...")
    with torch.no_grad(): 
        output_tensor = model(img_tensor)

    output_tensor = torch.clamp(output_tensor, 0, 1)

    output_tensor = output_tensor.squeeze(0)

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    save_image(output_tensor, args.output)
    
    result_img = Image.open(args.output)
    new_w, new_h = result_img.size
    print(f"Result is saved in: {args.output}")


if __name__ == '__main__':
    validate()