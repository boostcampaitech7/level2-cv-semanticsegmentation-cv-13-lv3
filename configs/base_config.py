class BaseConfig:
    def __init__(self):
        self.seed = 137
        self.epochs = 100
        self.lr = 1e-4
        self.batch_size = 16
        self.valid_batch_size = 16
        self.valid_interval = 2
        self.valid_thr = 0.5
        self.save_dir = "exp"
        self.save_name = "best_model.pt"
        self.image_root = "/data/ephemeral/home/data/train/DCM"
        self.label_root = "/data/ephemeral/home/data/train/outputs_json"
        self.encoder_name = "resnet34" 
        self.encoder_weights = "imagenet"
        self.classes = [
            'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
            'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
            'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
            'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
            'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
            'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
        ]