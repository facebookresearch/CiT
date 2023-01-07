# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
pre-configed sweeps.
"""

import json


class alltask_5k_mr005:
    batch_size = [1536], "bsz"
    max_update = [5000], "s"
    refilter = [100], "refilter"
    prefilter = [0.45], ""
    min_ratio = [0.05], "r"
    sublist = [True], ""
    val_task = [d for d in json.load(open("clipeval/dataset_catalog.json")).keys()], ""
    aug_tag = [True], ""
    nodes = [1], ""
    ngpus = [1], ""
