# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import json
import os
import pickle
import zipfile

import numpy as np
import torch
import random

from PIL import Image, ImageFile
from torchvision import datasets as t_datasets


ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def yfcc_loader(root, index):
    index = format(index, "0>8d")
    repo = index[:2]
    z = index[2: 5]
    file_img = index[5:] + '.jpg'
    path_zip = os.path.join(root, 'images', repo, z) + '.zip'
    with zipfile.ZipFile(path_zip, 'r') as myzip:
        img = Image.open(myzip.open(file_img))
    return img.convert('RGB')


def aug_tag(tag):
    delims = [" ", ",", ";", "/", "\n"]
    delim = random.choice(delims)[0]
    segs = [seg.strip() for seg in tag.split(",") if len(seg.strip()) > 0]
    random.shuffle(segs)
    tag = delim.join(segs)
    return tag


class ImageCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, args, dataset, root, metadata, task_example_ids=None, with_vision=True, with_text=True, max_sample=None):
        self.with_vision = with_vision
        self.with_text = with_text
        self.dataset = dataset
        self.root = root
        if hasattr(args, "aug_tag"):
            self.aug_tag = args.aug_tag
        if self.dataset in ["yfcc100m_tag"]:
            self.json_root = os.path.join(os.path.dirname(metadata), "yfcc100m_captioned_w_tag")
            self.samples = []
            if task_example_ids is not None:
                if isinstance(task_example_ids, list):
                    self.samples.extend(task_example_ids)
                else:
                    self.samples.extend(list(task_example_ids))
                print(f"apply task filter with {len(self.samples)} examples.")
            else:
                with open(metadata, 'rb') as f:
                    samples = pickle.load(f)
                self.samples.extend(samples)
                if max_sample is not None and len(self.samples) >= max_sample:
                    self.samples = self.samples[:max_sample]
        elif self.dataset in ['yfcc15m_tag', 'yfcc15m']:
            with open(metadata, 'rb') as f:
                samples = pickle.load(f)
            self.samples = []
            if task_example_ids is not None:
                if isinstance(task_example_ids, list):
                    # build the index of sample and follow the list order.
                    image_id_to_sample = {}
                    for image_id, title, desc, tag in samples:
                        title, desc, tag = title.strip(), desc.strip(), tag.strip()    
                        if len(title) > 0:
                            image_id_to_sample["_".join([str(image_id).zfill(8), "title"])] = {"image_id": image_id, "title": title}
                        if len(desc) > 0:
                            image_id_to_sample["_".join([str(image_id).zfill(8), "desc"])] = {"image_id": image_id, "desc": desc}
                        if "tag" in self.dataset and len(tag) > 0:
                            image_id_to_sample["_".join([str(image_id).zfill(8), "tag"])] = {"image_id": image_id, "tag": tag}
                    for image_key in task_example_ids:
                        if max_sample is not None and len(self.samples) >= max_sample:
                            break
                        image_id, field = image_key.split("_")
                        image_id = image_id.zfill(8)
                        image_key = "_".join([image_id, field])
                        self.samples.append(image_id_to_sample[image_key])
                else:
                    for image_id, title, desc, tag in samples:
                        title, desc, tag = title.strip(), desc.strip(), tag.strip()
                        if str(image_id).zfill(8) + "_title" in task_example_ids and len(title) > 0:
                            self.samples.append({"image_id": image_id, "title": title})
                        if str(image_id).zfill(8) + "_desc" in task_example_ids and len(desc) > 0:
                            self.samples.append({"image_id": image_id, "desc": desc})
                        if "tag" in self.dataset and str(image_id).zfill(8) + "_tag" in task_example_ids and len(tag) > 0:
                            self.samples.append({"image_id": image_id, "tag": tag})
                        if max_sample is not None and len(self.samples) >= max_sample:
                            break
                print(f"apply task filter with {len(self.samples)} examples.")
            else:
                for image_id, title, desc, tag in samples:
                    title, desc, tag = title.strip(), desc.strip(), tag.strip()
                    rec = {}
                    if len(title) > 0:
                        rec["title"] = title
                    if len(desc) > 0:
                        rec["desc"] = desc
                    if "tag" in self.dataset and len(tag) > 0:
                        rec["tag"] = tag
                    if len(rec) > 0:
                        rec["image_id"] = image_id
                        self.samples.append(rec)
                    if max_sample is not None and len(self.samples) >= max_sample:
                        break
        else:
            raise ValueError(f"unknown dataset {self.dataset}")

    def get_raw_item(self, i):
        if self.dataset in ["yfcc100m_tag"]:
            sample = self.samples[i]
            if isinstance(sample, str):
                index, key = sample.split("_")
            else:
                index = sample
                index = format(index, "0>8d")
            img = yfcc_loader(self.root, int(index)) if self.with_vision else None
            if self.with_text:
                repo = index[:2]
                z = index[2: 5]
                with open(f"{self.json_root}/{repo}/{z}.json") as fr:
                    repo_z = json.load(fr)
                    rec = repo_z[str(index).zfill(8)]
                if not isinstance(sample, str):
                    key = random.choice([key for key in rec if len(rec[key]) > 0])
                index = "_".join([str(index).zfill(8), key])
                if key == "tag" and (hasattr(self, "aug_tag") and self.aug_tag):
                    caption = aug_tag(rec[key])
                else:
                    caption = rec[key]
        elif self.dataset in ['yfcc15m_tag', 'yfcc15m']:
            rec = self.samples[i]
            index = rec["image_id"]
            img = yfcc_loader(self.root, index) if self.with_vision else None
            if self.with_text:
                key = random.choice([_key for _key in rec if _key != "image_id"])
                index = "_".join([str(index).zfill(8), key])
                if key == "tag" and hasattr(self, "aug_tag"):
                    caption = aug_tag(rec[key])
                else:
                    caption = rec[key]
        else:
            raise ValueError(f"unknown dataset {self.dataset}")
        return index, img, caption

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class ImageCaptionDatasetCLIP(ImageCaptionDatasetBase):
    def __init__(self, args, dataset, root, metadata, task_example_ids, transform=None, tokenizer=None, max_bert_length=77, with_vision=True, with_text=True, max_sample=None):
        super().__init__(args, dataset, root, metadata, task_example_ids, with_vision, with_text, max_sample)
        self.max_bert_length = max_bert_length
        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        index, img, caption = self.get_raw_item(i)
        result = {"image_ids": index}
        # apply transformation
        if img is not None and self.transform is not None:
            img = self.transform(img)
            result["pixel_values"] = img

        # tokenize caption
        if caption is not None and self.tokenizer is not None:
            inputs = self.tokenizer(caption, padding="max_length", truncation=True, max_length=self.max_bert_length, return_tensors="pt")
            for key in inputs:
                inputs[key] = inputs[key][0]
            result.update(**inputs)
            result["captions"] = caption
        return result


class FileListDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(images)
        self.labels = np.load(labels)

    def __getitem__(self, index):
        img = pil_loader(self.images[index])
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


def get_downstream_dataset(catalog, name, is_train, transform):
    entry = catalog[name]
    root = entry['path']
    if entry['type'] == 'imagefolder':
        dataset = t_datasets.ImageFolder(os.path.join(root, entry['train'] if is_train else entry['test']),
            transform=transform)
    elif entry['type'] == 'special':
        if name == 'cifar10':
            dataset = t_datasets.CIFAR10(root, train=is_train,
                transform=transform, download=True)
        elif name == 'cifar100':
            dataset = t_datasets.CIFAR100(root, train=is_train,
                transform=transform, download=True)
        elif name == 'stl10':
            dataset = t_datasets.STL10(root, split='train' if is_train else 'test',
                transform=transform, download=True)
        elif name == 'mnist':
            dataset = t_datasets.MNIST(root, train=is_train,
                transform=transform, download=True)
    elif entry['type'] == 'filelist':
        path = entry['train'] if is_train else entry['test']
        val_images = os.path.join(root, path + '_images.npy')
        val_labels = os.path.join(root, path + '_labels.npy')
        if name == 'clevr_counts':
            target_transform = lambda x: ['count_10', 'count_3', 'count_4', 'count_5', 'count_6', 'count_7', 'count_8', 'count_9'].index(x)
        else:
            target_transform = None
        dataset = FileListDataset(val_images, val_labels, transform, target_transform)
    else:
        raise Exception('Unknown dataset')

    return dataset
