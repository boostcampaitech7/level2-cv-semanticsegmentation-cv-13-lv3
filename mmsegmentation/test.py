import sys

sys.path.append('../lightningmodule')

from utils.utils import get_sorted_files_by_type, encode_mask_to_rle
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

def set_xraydataset(config):

    TEST_DATA_DIR = 'data/test'

    image_root = os.path.join(TEST_DATA_DIR, 'DCM')
    pngs = get_sorted_files_by_type(image_root, 'png')

    config.test_dataloader.dataset.image_files = np.array(pngs)
    config.test_dataloader.dataset.label_files = None

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


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
  
    cfg = set_xraydataset(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    cfg.test_evaluator['keep_results'] = True

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()

    rles, filename_and_class = runner.test_evaluator.metrics[0].get_results()

    inference(args, rles, filename_and_class)

if __name__ == '__main__':
    main()