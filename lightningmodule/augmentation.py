import albumentations as A

def load_transforms(args):
    transform = A.Compose([
        A.Resize(args.input_size, args.input_size),
        # A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0)
        # A.ColorJitter(brightness=0.1, contrast=0.4, saturation=0.005, hue=0.005, p=1.0)
    ])
    return transform