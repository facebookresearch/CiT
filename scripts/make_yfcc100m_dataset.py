# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pickle
import re
import time
import sqlite3
import webdataset as wds

from urllib.parse import unquote
from tqdm import tqdm


# Borrowed from SLIP but add tag field to be consistent with LiT: https://lbsn.vgiscience.org/yfcc-introduction/

def to_pkl():
    cleanhtml = re.compile('<a.*?>|</a>|<b>|</b>|<i>|</i>')
    cleanurl = re.compile('http\S+|www\S+')

    print('=> loading YFCC image ids')
    image_ids = np.load('data/yfcc15m/flickr_unique_ids.npy')
    image_ids = set(image_ids)

    print('=> loading CLIP image ids')
    print('=> collecting and cleaning subset captions')
    captioned = []
    valid_image_ids = []
    with open('/datasets01/yfcc100m/090517/yfcc100m_dataset.txt') as f:
        for l in tqdm(f):
            row = l.strip().split('\t')
            if int(row[0]) in image_ids:
                title = unquote(row[8]).replace('+', ' ')
                title = re.sub(cleanhtml, '', title)
                title = re.sub(cleanurl, '', title).strip()

                desc = unquote(row[9]).replace('+', ' ')
                desc = re.sub(cleanhtml, '', desc)
                desc = re.sub(cleanurl, '', desc).strip()

                tag = ",".join([row[10].strip(), row[11].strip()])
                tag = unquote(tag).replace('+', ' ')
                tag = re.sub(cleanhtml, '', tag)
                tag = re.sub(cleanurl, '', tag).strip()
                if any([len(title) > 0, len(desc) > 0, len(tag) > 0]):
                    captioned.append((int(row[0]), title, desc, tag))
                    valid_image_ids.append(int(row[0]))

    with open('data/yfcc100m/yfcc100m_captioned_w_tag.pkl', 'wb') as f:
        pickle.dump(captioned, f)

    with open('data/yfcc100m/yfcc100m_image_ids.pkl', 'wb') as f:
        pickle.dump(valid_image_ids, f)

    print('Total captioned images:', len(captioned))  # 94514285


def write_json():
    with open('data/yfcc100m/yfcc100m_captioned_w_tag.pkl', 'rb') as f:
        captioned = pickle.load(f)

    from collections import defaultdict
    repos = defaultdict(dict)

    for idx, (image_id, title, desc, tag) in enumerate(captioned):
        index = format(image_id, "0>8d")
        repo = index[:2]
        z = index[2: 5]
        repos[f"{str(repo).zfill(2)}_{str(z).zfill(3)}"][str(image_id).zfill(8)] = {"title": title, "desc": desc, "tag": tag}

    import json
    from pathlib import Path

    for repo in repos:
        _repo, z = repo.split("_")
        Path(f"data/yfcc100m/yfcc100m_captioned_w_tag/{_repo}").mkdir(parents=True, exist_ok=True)
        with open(f"data/yfcc100m/yfcc100m_captioned_w_tag/{_repo}/{z}.json", "w") as fw:
            json.dump(repos[repo], fw)


to_pkl()
write_json()
