import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import nn_arch.RRDBNet_arch as arch


def run_esrgan(input_folder: str,
               output_folder: str,
               mode: str = 'orig'):
    """
    Запускает ESRGAN для апскейла кадров.

    :param input_folder: папка с входными кадрами
    :param output_folder: папка для SR-кадров
    :param mode: режим esrgan модели (orig, rgb2, yuv)
    """

    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    os.makedirs(output_folder, exist_ok=True)

    print("\nRunning ESRGAN Super Resolution...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_path = f'./checkpoints/esrgan_{mode}.pth'

    if mode == 'orig':
        nf_base, gc_base = 64, 32
        print(f"Mode 'orig': number of channels (nf, gc) is halved (nf={nf_base}, gc={gc_base}).")
    elif mode == 'rgb2':
        nf_base, gc_base = 32, 16
        print(f"Mode 'rgb2': standart channels (nf={nf_base}, gc={gc_base}).")

    model = arch.RRDBNet(3, 3, nf=nf_base, nb=23, gc=gc_base)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    print(f'Model path {model_path}. \nTesting...')

    idx = 0
    for path in glob.glob(os.path.join(input_folder, '*')):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        print(idx, base)
        # read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        # img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        out_path = os.path.join(output_folder, f"{base}_sr.png")
        cv2.imwrite(out_path, output)

    print("ESRGAN finished successfully.")


if __name__ == '__main__':

    import sys
    from video_handler import video_to_photos, photos_to_video, cleanup_folders

    INPUT_FOLDER = './input_frames'
    UPSCALED_FRAMES = './frames_upscaled'

    if len(sys.argv) < 4:
        print("Usage: python script.py <input_video> <output_video> <mode>")
        sys.exit(1)

    input_video_path = sys.argv[1]
    output_video_path = sys.argv[2]
    video_mode = sys.argv[3]

    try:
        os.makedirs(INPUT_FOLDER, exist_ok=True)
        os.makedirs(UPSCALED_FRAMES, exist_ok=True)

        fps = video_to_photos(input_video_path)
        
        # Super Resolution (ESRGAN)
        print(f"\nRunning Super Resolution (ESRGAN mode: {video_mode})...")
        
        run_esrgan(input_folder=INPUT_FOLDER, 
                   output_folder=UPSCALED_FRAMES,
                   mode=video_mode)
        
        print("\nEncoding final video...")

        photos_to_video(output_video_path, fps, UPSCALED_FRAMES)
        
        print("\nProcess completed successfully!")
        
    except Exception as e:
        print(f"\nError during processing: {e}")

    finally:
        # Очистка временных файлов
        print("Cleaning up temporary frames...")
        cleanup_folders() 