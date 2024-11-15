from configs.config_factory import get_config
from utils.utils import get_sorted_files_by_type, set_seed
from utils.Gsheet import Gsheet_param
from xraydataset import XRayDataset, split_data

import torch.nn as nn
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import albumentations as A
from omegaconf import OmegaConf
from argparse import ArgumentParser
import os
import torch

def train_model(config):
    model_config = get_config(config['model_name'])
    seed_everything(config['seed'])
    set_seed(config['seed'])

    wandb_logger = WandbLogger(project=config['project_name'], name=config['run_name'], config=config)

    image_root = config['image_root']
    label_root = config['label_root']
    pngs = get_sorted_files_by_type(image_root, 'png')
    jsons = get_sorted_files_by_type(label_root, 'json')

    train_files, valid_files = split_data(pngs, jsons)

    train_dataset = XRayDataset(
        image_files=train_files['filenames'], 
        label_files=train_files['labelnames'], 
        transforms=A.Resize(config['input_size'], config['input_size'])
    )
    valid_dataset = XRayDataset(
        image_files=valid_files['filenames'], 
        label_files=valid_files['labelnames'], 
        transforms=A.Resize(config['input_size'], config['input_size'])
    )

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=config['valid_batch_size'],
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    criterion = nn.BCEWithLogitsLoss()
    seg_model = SegmentationModel(
        model_name=model_config.model_name,
        encoder_name=model_config.encoder_name,
        encoder_weights=model_config.encoder_weights,
        num_classes=model_config.classes,
        criterion=criterion,
        learning_rate=config['lr']
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['checkpoint_dir'],
        filename=f'{model_config.model_name}_best_model',
        monitor='val/dice',
        mode='max',
        save_top_k=3
    )

    trainer = Trainer(
        logger=wandb_logger,
        log_every_n_steps=5,
        max_epochs=config['max_epoch'],
        check_val_every_n_epoch=config['valid_interval'],
        callbacks=[checkpoint_callback],
        accelerator='gpu', 
        devices=1 if torch.cuda.is_available() else None
    )

    trainer.fit(seg_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    train_model(cfg)
    Gsheet_param(cfg)