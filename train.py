from configs.config_factory import get_config
from utils.Gsheet import Gsheet_param
from module.dataset import CustomDataModule
from module.trainer import CustomLightningModule

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from argparse import ArgumentParser
import torch


def train_model(config):
    model_config = get_config(config['model_name'])
    seed_everything(config['seed'])

    wandb_logger = WandbLogger(project=config['project_name'], name=config['run_name'], config=config)

    data_module = CustomDataModule(
        image_root=config['image_root'],
        label_root=config['label_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        transforms=config['transforms']
    )
    data_module.setup()

    seg_model = CustomLightningModule(
        model_name=config["model_name"],
        encoder_name=model_config.encoder_name,
        encoder_weights=model_config.encoder_weights,
        classes=model_config.classes,
        lr=config['lr']
    )


    checkpoint_callback = ModelCheckpoint(
        dirpath=config['checkpoint_dir'],
        filename=f'{model_config.model_name}_best_model',
        monitor='val_dice',
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

    trainer.fit(seg_model, datamodule=data_module)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration
    cfg = OmegaConf.load(args.config)

    # Train model
    train_model(cfg)

    # Log parameters to Gsheet
    Gsheet_param(cfg)