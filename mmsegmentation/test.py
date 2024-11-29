import sys

sys.path.append('../lightningmodule')

from utils import get_sorted_files_by_type
from xraydataset import split_data

# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

import numpy as np

from argparse import ArgumentParser

from mmengine.config import Config
from mmengine.runner import Runner

from omegaconf import OmegaConf

def set_xraydataset(args, config):

    image_files = None

    if args.valid:
        TRAIN_DATA_DIR = 'data/train'

        image_root = osp.join(TRAIN_DATA_DIR, 'DCM')
        label_root = osp.join(TRAIN_DATA_DIR, 'outputs_json')

        pngs = get_sorted_files_by_type(image_root, 'png')
        jsons = get_sorted_files_by_type(label_root, 'json')

        _, valid_files = split_data(pngs, jsons)

        image_files = np.array(pngs)
        #valid_files['filenames']
    else:
        TEST_DATA_DIR = 'data/test'

        image_root = osp.join(TEST_DATA_DIR, 'DCM')
        pngs = get_sorted_files_by_type(image_root, 'png')

        image_files = np.array(pngs)

    config.test_dataloader.dataset.image_files = image_files
    config.test_dataloader.dataset.label_files = None

    return config

def set_yaml_config_test(config, yaml_config):

    cfg_name = osp.splitext(osp.basename(yaml_config.config))[0]

    if yaml_config.tta is True:
        config.test_dataloader.dataset.pipeline = config.tta_pipeline
        config.tta_model.module = config.model
        config.model = config.tta_model

    ckpt_filepath = osp.join(yaml_config.checkpoint_dir, cfg_name, yaml_config.checkpoint_file.rsplit('.', 1)[0] + '.pth')
    config.load_from = ckpt_filepath
    
    return config

def disable_wandb(config):
    config.vis_backends = [dict(type='LocalVisBackend')]  
    config.visualizer = dict(type='SegLocalVisualizer', vis_backends=config.vis_backends, name='visualizer')
    return config

def test(args, yaml_cfg):

    cfg_name = osp.splitext(osp.basename(yaml_cfg.config))[0]
    # load config
    cfg = Config.fromfile(yaml_cfg.config)
    cfg = set_yaml_config_test(cfg, yaml_cfg)
    cfg = disable_wandb(cfg)
    cfg = set_xraydataset(args, cfg)

    cfg.launcher = 'none'

    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', cfg_name)

    # if args.show or args.show_dir:
    #     cfg = trigger_visualization_hook(cfg, args)

    # add output_dir in metric
    cfg.resume = False
    
    # build the runner from config
    runner = Runner.from_cfg(cfg)
    # start testing
    runner.test()

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs_cv13/base_config.yaml")
    parser.add_argument("--valid", action="store_true")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)  
    test(args, cfg)