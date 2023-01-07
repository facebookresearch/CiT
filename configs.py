# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import inspect

from collections import OrderedDict


class Config:
    dataset = "yfcc15m_tag"
    root = "data/yfcc15m"
    metadata = "data/yfcc15m/yfcc15m_w_tag.pkl"
    # data adaptation
    val_task = "imagenet"
    max_sample = None
    thres = 0.55
    num_workers = 6

    # model
    # model = "moco-bert"
    max_bert_length = 32
    trainable_weight = "head-all"
    vision_backbone = "moco"
    vision_pretrained = "pretrained_models/moco_hf"
    text_backbone = "bert"
    text_pretrained = "princeton-nlp/unsup-simcse-bert-base-uncased"
    output_root = "runs"

    # training
    fp16 = True
    lr = 5e-4
    warmup_div = 25
    min_lr = 1e-5
    weight_decay = 0.2
    head_weight_decay = 1.
    device = "cuda"
    dist_eval = False
    accum_iter = 1
    eval = False
    pin_mem = False
    resume = None
    clip_grad = None

    loss = "CiTCLIPLossGrad"

    curate = 0

    # evaluate
    use_template = True
    patience = None
    eval_steps = 500
    seed = 0
    dist_on_itp = False
    log_dir = None

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        if not hasattr(self, "warmup_steps"):
            self.warmup_steps = int(self.max_update / self.warmup_div)  # TODO move this to main?
        if not hasattr(self, "output_dir"):
            self.output_dir = inspect.stack()[1][3]
        self.output_dir = os.path.join(self.output_root, self.output_dir)
        print("config.output_dir =", self.output_dir)

    def add_cmd_args(self, cmd_args):
        for key, value in vars(cmd_args).items():
            if not key.startswith("__") and value is not None:
                setattr(self, key, value)
        return self

    def __str__(self):
        return "\n".join([f"{k}={v}" for k, v in vars(self).items()])


def build_from_sweep_config(sweep_config):
    sweep_dict = OrderedDict()
    key_to_short = OrderedDict()
    key_to_card = OrderedDict()
    sweep_name = sweep_config.__name__
    cards = 1
    for key, value in vars(sweep_config).items():
        if not key.startswith("__"):
            sweep_dict[key] = value[0] if isinstance(value, tuple) else value
            cards *= len(sweep_dict[key])
            key_to_card[key] = len(sweep_dict[key])
            key_to_short[key] = value[1] if isinstance(value, tuple) else ""

    all_update_dicts = []
    for sweep_idx in range(cards):
        key_to_idx = OrderedDict()
        for key in key_to_card:
            key_to_idx[key] = sweep_idx % key_to_card[key]
            sweep_idx = sweep_idx // key_to_card[key]
        update_dict = OrderedDict()
        for key, idx in key_to_idx.items():
            update_dict[key] = sweep_dict[key][idx]
        update_dict["output_dir"] = "_".join([value+str(update_dict[key]).replace("/", ".") for key, value in key_to_short.items()])
        update_dict["output_dir"] = os.path.join(sweep_name, update_dict["output_dir"])
        all_update_dicts.append(update_dict)

    assert len(all_update_dicts) == cards
    return all_update_dicts
