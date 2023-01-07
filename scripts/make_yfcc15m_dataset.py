# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved


import numpy as np
import pickle
import re
from urllib.parse import unquote
from tqdm import tqdm


# Borrowed from SLIP but add tag field to be consistent with LiT: https://lbsn.vgiscience.org/yfcc-introduction/

cleanhtml = re.compile('<a.*?>|</a>|<b>|</b>|<i>|</i>')
cleanurl = re.compile('http\S+|www\S+')

print('=> loading YFCC image ids')
image_ids = np.load('data/yfcc15m/flickr_unique_ids.npy')
image_ids = set(image_ids)

print('=> loading CLIP image ids')
clip_ids = set()
with open('data/yfcc15m/yfcc100m_subset_data.tsv') as f:
    for l in tqdm(f.readlines()):
        row = l.strip().split('\t')
        clip_ids.add(int(row[0]))

print('=> collecting and cleaning subset captions')
captioned = []

with open('/datasets01/yfcc100m/090517/yfcc100m_dataset.txt') as f:
    for l in tqdm(f):
        row = l.strip().split('\t')
        if int(row[0]) in image_ids:
            if int(row[0]) in clip_ids:
                title = unquote(row[8]).replace('+', ' ')
                title = re.sub(cleanhtml, '', title)
                title = re.sub(cleanurl, '', title)

                desc = unquote(row[9]).replace('+', ' ')
                desc = re.sub(cleanhtml, '', desc)
                desc = re.sub(cleanurl, '', desc)

                tag = ",".join([row[10].strip(), row[11].strip()])
                tag = unquote(tag).replace('+', ' ')
                tag = re.sub(cleanhtml, '', tag)
                tag = re.sub(cleanurl, '', tag)

                captioned.append((int(row[0]), title, desc, tag))

with open('data/yfcc15m/yfcc15m_w_tag.pkl', 'wb') as f:
    pickle.dump(captioned, f)

print('Total captioned images:', len(captioned))  # 14689580
