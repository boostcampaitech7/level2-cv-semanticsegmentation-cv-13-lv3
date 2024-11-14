import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import albumentations as A
import pandas as pd
import numpy as np
import argparse
import os
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from github.dataset_eda import XRayInferenceDataset 
from github.eda.utils import encode_mask_to_rle, set_seed

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

def inference(args):
    set_seed(args.seed)

    model = smp.DeepLabV3Plus(
        encoder_name=args.encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=len(CLASSES)
    )
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_file)))
    model = model.cuda()
    model.eval()

    test_transform = A.Resize(args.input_size, args.input_size)
    test_dataset = XRayInferenceDataset(image_root=args.test_image_root, transforms=test_transform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )


    rles = []
    filename_and_class = []
    with torch.no_grad():
        for images, image_names in tqdm(test_loader, total=len(test_loader), desc="Running Inference"):
            images = images.cuda()
            outputs = model(images)
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > args.test_threshold).cpu().numpy()

            for output, image_name in zip(outputs, image_names):
                for i, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{CLASSES[i]}_{image_name}")

    classes, filenames = zip(*[fc.split("_") for fc in filename_and_class])
    image_names = [os.path.basename(f) for f in filenames]
    
    df = pd.DataFrame({
        "image_name": image_names,
        "class": classes,
        "rle": rles,
    })
    df.to_csv(args.output_csv, index=False)
    print(f"Inference results saved to {args.output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add missing arguments for model encoder
    parser.add_argument('--encoder_name', type=str, default='tu-xception71', help="encoder name for model")
    parser.add_argument('--encoder_weights', type=str, default=None, help="pre-trained weights for encoder (default: None)")

    parser.add_argument('--seed', type=int, default=137, help='random seed')
    parser.add_argument('--checkpoint_dir', type=str, default="/data/ephemeral/home/exp", help="Directory where model is saved")
    parser.add_argument('--checkpoint_file', type=str, default="best.pt", help="Model checkpoint file name")
    parser.add_argument('--test_image_root', type=str, default="/data/ephemeral/home/test/DCM", help="Directory with test images")
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for inference')
    parser.add_argument('--test_threshold', type=float, default=0.5, help='Threshold for binarizing segmentation output')
    parser.add_argument('--input_size', type=int, default=1024, help="Input size for resizing during inference")
    parser.add_argument('--output_csv', type=str, default="output.csv", help="Output CSV file name for predictions")

    args = parser.parse_args()
    inference(args)