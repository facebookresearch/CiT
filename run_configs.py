# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from configs import Config


########### running ###########
# torchrun --nproc_per_node=8 main.py <config>

def eval_yfcc15m_in1k_mocob16():
    return Config(
        output_dir="yfcc15m_in1k_mocob16",
        eval=True,
        resume="checkpoint-best.pth",
        dataset="yfcc15m_tag",
        metadata="data/yfcc15m/yfcc15m_w_tag.pkl",
        root="data/yfcc15m",
        trainable_weight="head-all",
        batch_size=1024,
        max_bert_length=32,
        max_update=5000,
        weight_decay=0.2,
        head_weight_decay=1.,
        eval_steps=500,
        curate=100,
        min_ratio=0.003,
        extra_prompt=True,
        aug_tag=True,
        nodes=1, ngpus=1,
    )


def yfcc15m_in1k_mocob16():
    return Config(
        val_task="imagenet",
        dataset="yfcc15m_tag",
        metadata="data/yfcc15m/yfcc15m_w_tag.pkl",
        root="data/yfcc15m",
        trainable_weight="head-all",
        batch_size=1024,
        max_bert_length=32,
        max_update=5000,
        weight_decay=0.2,
        head_weight_decay=1.,
        eval_steps=500,
        curate=100,
        min_ratio=0.003,
        extra_prompt=True,
        aug_tag=True,
        nodes=2, ngpus=8,
    )


def yfcc100m_in1k_mocob16():
    return Config(
        val_task="imagenet",
        dataset="yfcc100m_tag",
        metadata="data/yfcc100m/yfcc100m_image_ids.pkl",
        root="/datasets01/yfcc100m/090517",
        trainable_weight="head-all",
        batch_size=1024,
        max_bert_length=32,
        max_update=5000,
        weight_decay=0.2,
        head_weight_decay=1.,
        eval_steps=500,
        curate=100,
        thres=0.7,
        sublist=True,
        min_ratio=0.01,
        extra_prompt=True,
        aug_tag=True,
        nodes=2, ngpus=8,
    )
