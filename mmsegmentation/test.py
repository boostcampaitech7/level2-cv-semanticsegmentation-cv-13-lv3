import sys

sys.path.append('../lightningmodule')

from omegaconf import OmegaConf
from utils.utils import get_sorted_files_by_type, encode_mask_to_rle
from constants import IND2CLASS

# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from tools.test import parse_args, trigger_visualization_hook

import numpy as np
import pandas as pd

from argparse import Namespace

def set_xraydataset(config):

    TEST_DATA_DIR = 'data/test'

    image_root = os.path.join(TEST_DATA_DIR, 'DCM')
    pngs = get_sorted_files_by_type(image_root, 'png')

    config.test_dataloader.dataset.image_files = np.array(pngs)
    config.test_dataloader.dataset.label_files = None

    return config

def set_yaml_config_test(config, yaml_config):

    cfg_name = osp.splitext(osp.basename(yaml_config.config))[0]

    if yaml_config.tta is True:
        config.test_dataloader.dataset.pipeline = config.tta_pipeline
        config.tta_model.module = config.model
        config.model = config.tta_model

    ckpt_filepath = os.path.join(yaml_config.checkpoint_dir, cfg_name, yaml_config.checkpoint_file.rsplit('.', 1)[0] + '.pth')
    config.load_from = ckpt_filepath
    
    return config

def inference(args, rles, filename_and_class, thr=0.5):

    classes, filename = zip(*[x.split("_") for x in filename_and_class])

    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame(
        {
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        }
    )

    df.to_csv("output.csv", index=False)


def test(args, yaml_cfg):

    cfg_name = osp.splitext(osp.basename(yaml_cfg.config))[0]
    # load config
    cfg = Config.fromfile(yaml_cfg.config)
  
    cfg = set_xraydataset(cfg)
    cfg = set_yaml_config_test(cfg, yaml_cfg)

    cfg.launcher = args.launcher

    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', cfg_name)

    # if args.show or args.show_dir:
    #     cfg = trigger_visualization_hook(cfg, args)

    # add output_dir in metric
    cfg.test_evaluator['keep_results'] = True

    cfg.resume = False
    
    # build the runner from config
    runner = Runner.from_cfg(cfg)
    # start testing
    runner.test()

    rles, filename_and_class = runner.test_evaluator.metrics[0].get_results()

    inference(args, rles, filename_and_class)

if __name__ == '__main__':
    args = parse_args()
    yaml_config_path = './configs_cv13/base_config.yaml'
    with open(yaml_config_path, 'r') as f:
        cfg = OmegaConf.load(f)  
    test(args, cfg)