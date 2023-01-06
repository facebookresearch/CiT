# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import datetime
import numpy as np
import os
import time
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from collections import defaultdict

import losses

import util.misc as misc

from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models_citclip import build_model
from engine import train_one_epoch, evaluate, iterative_classcurate
from weights import freeze_model


def get_mean_std(args):
    if "augreg" in args.vision_backbone or "augreg" in args.vision_pretrained:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    return mean, std


def get_val_transform(args):
    """moved from SLIP's eval_zeroshot.py"""
    import torchvision.transforms as transforms
    mean, std = get_mean_std(args)
    print(args.vision_backbone, "val_normalizer", mean, std)
    return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
    ])


def get_train_transform(args):
    import torchvision.transforms as transforms
    trans = [transforms.RandomResizedCrop(224, scale=(0.5, 1.0))]
    if hasattr(args, "inmem") and args.inmem:  # use in-mem training / no dataloader for consumer dataset.
        from torchvision.transforms.functional import pil_to_tensor
        trans.append(pil_to_tensor)
    else:
        trans.append(transforms.ToTensor())
        mean, std = get_mean_std(args)
        print(args.vision_backbone, "train_normalizer", mean, std)
        trans.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(trans)


def build_dataset(args, tokenizer):
    from clipeval import datasets
    train_transform = get_train_transform(args)
    train_task_example_ids = None
    if hasattr(args, "pcurate") or (args.val_task is not None and args.curate == 0):   # no validation for full yfcc15m training (same as SLIP/CLIP).
        thres = args.pcurate if hasattr(args, "pcurate") else args.thres
        if args.dataset in ["yfcc15m_tag"]:
            task_meta = torch.load(f"data/CLIP/{args.dataset}/{args.val_task}_ub_{args.dataset}_simcse{thres}_{args.max_bert_length}.pt")
            if hasattr(args, "sublist") and args.sublist:
                train_task_example_ids = task_meta["example_ids"]
            else:
                train_task_example_ids = set(task_meta["example_ids"])
            print("train_task_example_ids_key", len(train_task_example_ids))
        else:
            task_meta = torch.load(f"data/CLIP/CLIP_eval/{args.val_task}_ub_{args.dataset}_simcse{thres}.pt")
            if hasattr(args, "sublist") and args.sublist:
                train_task_example_ids = task_meta["example_ids"]
            else:
                train_task_example_ids = set(task_meta["example_ids"])
            print("train_task_example_ids", len(train_task_example_ids))
    tar_files = None
    train_dataset = datasets.ImageCaptionDatasetCLIP(
        args, args.dataset, args.root, args.metadata, train_task_example_ids,
        train_transform, tokenizer, args.max_bert_length, max_sample=args.max_sample
    )
    return train_dataset, None, train_transform, tar_files


def producer_collator(batch_list):
    result = defaultdict(list)
    for item in batch_list:
        for key in item:
            if key not in ["__key__"]:
                result[key].append(item[key])
    for key in result:
        if key not in ["image_ids", "__key__", "captions"]:
            result[key] = torch.stack(result[key])
    return result


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model, tokenizer = build_model(args)
    model = freeze_model(model, args)
    model.to(device)
    dataset_train, dataset_val, train_transform, tar_files = build_dataset(args, tokenizer)
    val_transform = get_val_transform(args)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    ) if not isinstance(dataset_train, torch.utils.data.IterableDataset) else None
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
    else:
        sampler_val = None if dataset_val is None else torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    if not isinstance(dataset_train, torch.utils.data.IterableDataset):
        print("len(dataset)", len(dataset_train))
    else:
        print("cannot estimate len of torch.utils.data.IterableDataset.")

    if args.distributed:
        find_unused = False
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=find_unused)
        model_without_ddp = model.module

    # https://github.com/rwightman/pytorch-image-models/blob/fd360ac951a179474917f4b2d21db8669bf87f68/timm/models/vision_transformer.py#L407
    no_weight_decay_list = {'pos_embed', 'cls_token', 'dist_token'}  # THIS DOESN'T MATTER YET as we frozen all.
    head_weight_decay_list = {"visual_projection", "text_projection"}

    p_wd, p_no_wd = [], []
    p_head_wd = []
    # only apply 1-dim no decay for now.
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim == 1 or n in no_weight_decay_list:
            p_no_wd.append(p)
        elif hasattr(args, "no_wd_emb") and isinstance(p, torch.nn.Embedding):
            p_no_wd.append(p)
        elif hasattr(args, "no_wd_ln") and isinstance(p, torch.nn.LayerNorm):
            p_no_wd.append(p)
        elif hasattr(args, "head_weight_decay") and [True for _part in head_weight_decay_list if _part in n]:
            p_head_wd.append(p)
        else:
            p_wd.append(p)

    param_groups = [{"params": p_wd, "weight_decay": args.weight_decay},
                    {"params": p_no_wd, "weight_decay": 0.}]

    if p_head_wd:
        param_groups.append({"params": p_head_wd, "weight_decay": args.head_weight_decay})

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, eps=1e-8)
    loss_scaler = NativeScaler(args.fp16)

    start_epoch, best_acc, step = 0, [0.], [0]
    if args.resume:
        if args.resume.endswith(".pth"):  # a pytorch checkpoint for resuming training.
            if args.resume.startswith("checkpoint"):
                args.resume = os.path.join(args.output_dir, args.resume)
            start_epoch, _, best_acc, step = misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
            best_acc, step = [best_acc], [step if step is not None else 0]

            if isinstance(dataset_train, torch.utils.data.IterableDataset):
                # random from step to avoid dupped train.
                dataset_train.start_shard_id = step[0] % dataset_train.num_shards
            print("resuming", args.resume, "from step", step[0], "with best_acc", best_acc[0])
        else:
            print("assuming a huggingface transformer pretrained model (no optimizer states).")
            from models_citclip import CiTCLIPVisionTextDualEncoderModel
            metric = evaluate(args, model, val_transform, tokenizer)
            model = CiTCLIPVisionTextDualEncoderModel.from_pretrained(args.resume)
    if args.eval:
        metric = evaluate(args, model, val_transform, tokenizer)
        json_str = json.dumps({"step": step[0], "acc": metric, "seen": eff_batch_size * step[0]})
        print(json_str)
        exit(0)

    criterion = getattr(losses, args.loss)().to(device)

    print("criterion = %s" % str(criterion))

    if args.curate is not None and args.curate > 1:
        curate_batch_size = args.batch_size * 2

        dataset_train.with_vision = True if hasattr(args, "inmem") and args.inmem else False
        data_loader_producer = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=curate_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            collate_fn=producer_collator,
            persistent_workers=True
        )

        def producer_fn(epoch):
            while True:
                data_loader_producer.sampler.set_epoch(epoch)
                for batch in data_loader_producer:
                    yield batch
                epoch += 1

        producer_iter = iter(producer_fn(start_epoch))

    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

    data_loader_val = None if dataset_val is None else torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    import math

    if not isinstance(dataset_train, torch.utils.data.IterableDataset) and not args.curate:
        epochs = math.ceil(args.max_update / (len(dataset_train) // eff_batch_size))
        print(f"Start training for {args.max_update} steps / {epochs} epochs")
    else:
        epochs = 1000000  # a big number to allow infinity run on iterativedataset.
        print(f"Start training for {args.max_update} steps on torch.utils.data.IterableDataset or curate dataset, the checkpoint is stateless.")
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        if step[0] >= args.max_update:
            break

        if args.curate is not None and (args.curate > 0 and step[0] % args.curate == 0):
            curate_cls = iterative_classcurate
            all_example_ids = curate_cls(step, device, producer_iter, model, tokenizer, args)
            print(len(all_example_ids), "after curate", args.curate * args.batch_size, "expected")
            if hasattr(args, "inmem") and args.inmem:
                data_loader_train = all_example_ids
            else:
                if hasattr(args, "sublist") and args.sublist:
                    assert isinstance(all_example_ids, list)
                    all_example_ids = all_example_ids[:args.curate * args.batch_size]
                else:
                    all_example_ids = set(list(all_example_ids)[:args.curate * args.batch_size])
                assert len(all_example_ids) == args.curate * args.batch_size
                from clipeval import datasets
                dataset_train = datasets.ImageCaptionDatasetCLIP(args,
                    args.dataset, args.root, args.metadata, all_example_ids,
                    train_transform, tokenizer, args.max_bert_length, max_sample=args.max_sample
                )
                data_loader_train = torch.utils.data.DataLoader(
                    dataset_train, shuffle=True,  # just a local sampler.
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem,
                    drop_last=True,
                )

        if hasattr(data_loader_train, "sampler") and isinstance(data_loader_train.sampler, torch.utils.data.DistributedSampler):
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, model_without_ddp, criterion, tokenizer, data_loader_train, data_loader_val, val_transform, best_acc,
            optimizer, device, epoch, step, loss_scaler, eff_batch_size,
            args.clip_grad,
            log_writer=log_writer,
            args=args
        )

        if not isinstance(dataset_train, torch.utils.data.IterableDataset):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, epoch_name="last", best_acc=best_acc[0], step=step[0])
        else:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=0, epoch_name="last", best_acc=best_acc[0], step=step[0])

    # if log_writer is not None:
    #     log_writer.finish()
    args.resume = os.path.join(args.output_dir, "checkpoint-best.pth")
    if os.path.isfile(args.resume):
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    metric = evaluate(args, model, val_transform, tokenizer)
    json_str = json.dumps({"step": step[0], "acc": metric, "seen": eff_batch_size * step[0]})
    print(json_str)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    '''see configs.py or sweep.py (we only allow pre-defined config).'''
    parser = argparse.ArgumentParser(description='CiTCLIP', add_help=False)
    parser.add_argument('config_name', type=str, help='see configs.py')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--eval', default=None, action='store_true')

    cmd_args = parser.parse_args()

    import run_configs
    config = getattr(run_configs, cmd_args.config_name)().add_cmd_args(cmd_args)
    return config


if __name__ == '__main__':
    args = parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
