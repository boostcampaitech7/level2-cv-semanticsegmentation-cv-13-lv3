import sys

sys.path.append('../lightningmodule')

from omegaconf import OmegaConf
from utils import get_sorted_files_by_type, encode_mask_to_rle
from constants import IND2CLASS

sys.path.append('../mmseg/')

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

    TEST_DATA_DIR = 'data/train'

    image_root = os.path.join(TEST_DATA_DIR, 'DCM')
    pngs = get_sorted_files_by_type(image_root, 'png')

    config.test_dataloader.dataset.image_files = np.array(pngs)
    config.test_dataloader.dataset.label_files = None

    return config

def set_yaml_config_test(config, yaml_config):

    if yaml_config.tta is True:
        config.test_dataloader.dataset.pipeline = config.tta_pipeline
        config.tta_model.module = config.model
        config.model = config.tta_model

    ckpt_filepath = os.path.join(yaml_config.checkpoint_dir, osp.basename(yaml_config.checkpoint_file)[0] + 'pth')
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
    # load config
    cfg = Config.fromfile(args.config)
  
    cfg = set_xraydataset(cfg)
    cfg = set_yaml_config_test(cfg, yaml_cfg)

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # add output_dir in metric
    cfg.test_evaluator['keep_results'] = True

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