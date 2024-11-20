import sys

sys.path.append('../lightningmodule')

from omegaconf import omegaconf
from utils import get_sorted_files_by_type, set_seed, Gsheet_param
from xraydataset import split_data

# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS

from tools.train import parse_args
from test import test

def set_xraydataset(config):
    TRAIN_DATA_DIR = 'data/train'

    image_root = os.path.join(TRAIN_DATA_DIR, 'DCM')
    label_root = os.path.join(TRAIN_DATA_DIR, 'outputs_json')

    pngs = get_sorted_files_by_type(image_root, 'png')
    jsons = get_sorted_files_by_type(label_root, 'json')

    train_files, valid_files = split_data(pngs, jsons)

    config.train_dataloader.dataset.image_files = train_files['filenames']
    config.train_dataloader.dataset.label_files = train_files['labelnames']

    config.val_dataloader.dataset.image_files = valid_files['filenames']
    config.val_dataloader.dataset.label_files = valid_files['labelnames']

    return config

def set_yaml_cfg(config, yaml_config):

    set_seed(yaml_config.seed)

    args_dict = OmegaConf.to_container(yaml_config, resolve=True)
    run_name = args_dict.pop('run_name', None)
    project_name = args_dict.pop('project_name', None)
    wandb_id = args_dict.pop('wandb_id', None)

    config.resume = yaml_config.resume

    config.visualizer.vis_backends[1].init_kwargs.project=project_name
    config.visualizer.vis_backends[1].init_kwargs.name=run_name
    config.visualizer.vis_backends[1].init_kwargs.resume='must' if wandb_id != -1 else 'allow'
    config.visualizer.vis_backends[1].init_kwargs.id=wandb_id if wandb_id != -1 else None
    config.visualizer.vis_backends[1].init_kwargs.config=args_dict

    config.train_dataloader.batch_size = yaml_config.batch_size
    config.train_dataloader.num_workers = yaml_config.num_workers
    config.train_dataloader.dataset.pipeline[2].scale = (yaml_config.input_size, yaml_config.input_size) # Resize

    config.optim_wrapper.optimizer.lr=yaml_config.lr
    # enable automatic-mixed-precision training
    if yaml_config.amp is True:
        optim_wrapper = config.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            config.optim_wrapper.type = 'AmpOptimWrapper'
            config.optim_wrapper.loss_scale = 'dynamic'

    config.max_iters=yaml_config.max_step
    config.param_scheduler[0].end=config.max_iters

    config.train_cfg.max_iters=config.max_iters
    config.train_cfg.val_interval=yaml_config.valid_step

    config.default_hooks.checkpoint.interval=yaml_config.valid_step
    config.default_hooks.checkpoint.out_dir=yaml_config.checkpoint_dir
    config.default_hooks.checkpoint.filename_tmpl=yaml_config.checkpoint_file.rsplit('.', 1)[0] + "_{}.pth"

    return config

def train(yaml_cfg):
    # load config
    cfg = Config.fromfile(yaml_cfg.config)

    cfg = set_yaml_cfg(cfg, yaml_cfg)
    cfg = set_xraydataset(cfg)

    cfg.launcher = 'none'

    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(yaml_cfg.config))[0])

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    yaml_config_path = './configs_cv13/base_config.yaml'
    with open(yaml_config_path, 'r') as f:
        cfg = OmegaConf.load(f)  
    train(cfg)
    Gsheet_param(cfg)

    # if cfg.auto_eval is True:
    #     test(cfg)
