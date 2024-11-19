import albumentations as A

def load_augmentation(args):
    transform = A.Compose([
        A.Resize(args.input_size, args.input_size),
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1),
    ])
    return transform