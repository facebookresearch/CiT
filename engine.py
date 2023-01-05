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

import math
import sys
import json

import torch

import util.misc as misc
import util.lr_sched as lr_sched

from typing import Iterable
from collections import defaultdict


def to_device(samples, device, args):
    inputs = {}
    for key in samples:
        if key not in ["image_ids", "captions", "__key__"]:
            inputs[key] = samples[key].to(device, non_blocking=True)
            if key == "pixel_values" and inputs[key].dtype == torch.uint8:
                from main import get_mean_std
                # inmem data. normalize it.
                inputs[key] = inputs[key].to(torch.float32).div_(255.)  # b, 3, 224, 224
                mean, std = get_mean_std(args)
                mean = torch.as_tensor(mean, device=inputs[key].device)[None, :, None, None]
                std = torch.as_tensor(std, device=inputs[key].device)[None, :, None, None]
                inputs[key] = inputs[key].sub_(mean).div_(std)
    return inputs


@torch.no_grad()
def evaluate(args, model, val_transform, tokenizer):
    from clipeval import datasets, eval_zeroshot

    catalog, all_templates, all_labels = eval_zeroshot.load_metadata("clipeval")

    if args.val_task is None or args.val_task in ["mt", "imagenet21k", "imagenet1k"]:  # infer val_task for multitasking.
        val_task = "imagenet"
    else:
        val_task = args.val_task

    metrics = {}
    for d in catalog:  # assume multitask on CLIP suite by default and early stop if IN only.
        if not args.eval and d != val_task:  # training only eval on val_task.
            continue
        if args.eval and args.val_task not in ["mt", "imagenet21k", "imagenet1k"] and d != val_task:
            continue
        val_dataset = datasets.get_downstream_dataset(
            catalog, d, is_train=False, transform=val_transform)
        templates = all_templates[d]
        labels = all_labels[d]

        if args.val_task not in ["mt", "imagenet21k", "imagenet1k"] and (hasattr(args, "extra_prompt") and args.extra_prompt) and d == "imagenet":  # not eval MT in LiT setup.
            templates.extend(["A photo of a {}", "{}"])  # see LiT page 16.

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size//2, shuffle=False,
            num_workers=args.num_workers, pin_memory=False, drop_last=False)

        if not args.use_template:
            templates = ["{}"]

        metric = eval_zeroshot.evaluate(d, val_loader, templates, labels, model, tokenizer, args.max_bert_length, False)
        metrics[d] = metric
        if args.eval:
            json_str = json.dumps({"task": d, "acc": metric})
            misc.print_json(args.output_dir, json_str)
    return metrics if len(metrics) > 1 else metrics[val_task]  # be compatible for ImageNet only evaluation.


def append_dataset(dataset, batch, mask_selector, batch_size):
    if "pixel_values" in batch:
        assert batch["pixel_values"].dtype == torch.uint8
    if mask_selector.sum().item() == 0:
        return
    assert len(dataset[-1]["image_ids"]) <= batch_size
    if len(dataset[-1]["image_ids"]) == batch_size:
        dataset.append(defaultdict(list))
    batch_len = len(batch["image_ids"])
    for key in batch:
        assert batch_len == len(batch[key])
        for ix, selected in enumerate(mask_selector):
            if selected:
                dataset[-1][key].append(batch[key][ix])

    while len(dataset[-1]["image_ids"]) >= batch_size:
        last_batch = dataset[-1]
        new_batch = {}
        for key in last_batch:
            value = last_batch[key]
            if len(value) >= batch_size:
                last_batch[key] = value[:batch_size]
                if torch.is_tensor(value[0]):
                    last_batch[key] = torch.stack(last_batch[key])
                if len(value) > batch_size:
                    new_batch[key] = value[batch_size:]
        if new_batch:
            dataset.append(new_batch)
        else:
            return


def train_one_epoch(model: torch.nn.Module, model_without_ddp, criterion: torch.nn.Module, tokenizer,
                    data_loader: Iterable, data_loader_val: Iterable, val_transform, best_acc, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, step, loss_scaler, eff_batch_size, max_norm: float = 0,
                    # mixup_fn: Optional[Mixup] = None,
                    log_writer=None,
                    args=None):
    model.train(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    # assuming data_loader is either a real dataloader or inmem as a list of batches?
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header, args.max_update)):
        if step[0] > args.max_update:
            break

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_step_learning_rate(optimizer, step[0], args.lr, args.min_lr, args.warmup_steps, args.max_update)

        inputs = to_device(samples, device, args)

        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(**inputs)
            loss = criterion(**outputs)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        update_grad = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=update_grad)

        if update_grad:
            step[0] += 1
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            log_writer.add_scalar('lr', max_lr, step[0])
            log_writer.add_scalar('loss', loss_value_reduce, step[0])

        if step[0] and step[0] % args.eval_steps == 0:
            metric = evaluate(args, model, val_transform, tokenizer)
            json_str = json.dumps({"step": step[0], "acc": metric, "seen": eff_batch_size * step[0]})
            misc.print_json(args.output_dir, json_str)
            if log_writer is not None:
                log_writer.add_scalar('acc', metric, step[0])

            if isinstance(data_loader, list) or (hasattr(data_loader, "dataset") and isinstance(data_loader.dataset, torch.utils.data.IterableDataset)):
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=0, epoch_name="last", best_acc=best_acc[0], step=step[0])
            if metric > best_acc[0]:
                best_acc[0] = metric
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=step[0], epoch_name="best", best_acc=best_acc[0], step=step[0])
            model.train(True)

        if step[0] and curate_condition(step[0], args):
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def curate_condition(step, args):
    if args.curate and step % args.curate == 0:
        return True
    else:
        return False


def curate_scheduler(step, args):
    return args.curate


def max_sim(logits, thres):
    logits, idx = logits.max(dim=-1)
    return logits > thres, idx


ratio = 1.0
thres = None


def thres_scheduler(step, args):
    return args.thres


def while_condition(example_ids, step, args):
    if hasattr(args, "inmem") and args.inmem:
        return len(example_ids) < curate_scheduler(step, args) or (len(example_ids) == curate_scheduler(step, args) and len(example_ids[-1]["image_ids"]) < args.batch_size)
    else:
        return len(example_ids) < (curate_scheduler(step, args) * args.batch_size)


@torch.no_grad()
def iterative_classcurate(step, device, producer_iter, model, tokenizer, args):
    model.eval()
    from clipeval import eval_zeroshot

    catalog, all_templates, all_labels = eval_zeroshot.load_metadata("clipeval")
    if args.val_task == "mt":
        labels = set()
        for d in catalog:
            for label in all_labels[d]:
                if isinstance(label, list):
                    for _label in label:
                        labels.add(_label)
                else:
                    labels.add(label)
        labels = list(labels)
    elif args.val_task == "imagenet21k":
        labels = set()
        with open("clipeval/imagenet21k_wordnet_lemmas.txt", "r") as fr:
            for line in fr:
                labels.add(line.strip())
        labels = list(labels)
    else:
        d = args.val_task  # infer catalog_subsets
        labels = all_labels[d]

    templates = ["{}"] if not (hasattr(args, "templatefilter") and args.templatefilter) else all_templates[args.val_task] # no templates for now.

    labels_emb = []
    with torch.cuda.amp.autocast():
        labels_emb, _, _ = eval_zeroshot.build_text_features(
            templates, labels, model, tokenizer, args.max_bert_length, skip_text_projection=True)
    labels_emb = labels_emb.t().to(torch.float32)
    if hasattr(args, "sublist") and args.sublist:
        example_ids = []
    else:
        example_ids = set()
    total_example = 0
    global thres
    thres = thres_scheduler(step[0], args)
    while while_condition(example_ids, step[0], args):
        samples = next(producer_iter)
        image_ids = samples["image_ids"]
        total_example += len(image_ids)
        if hasattr(args, "skip_step") and step[0] < args.skip_step:
            mask_selector = torch.ones((len(image_ids),), dtype=torch.bool)
        else:
            inputs = to_device(samples, device, args)
            with torch.cuda.amp.autocast():
                text_embeds = model(**inputs, skip_text_projection=False if hasattr(args, "project_emb") else True)["text_embeds"]
            text_embeds = text_embeds.to(torch.float32)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            logits = torch.matmul(text_embeds, labels_emb).cpu()
            mask_selector, class_idx = max_sim(logits, thres)
        batch_ratio = float(mask_selector.sum() / len(mask_selector))
        if hasattr(args, "min_ratio") and batch_ratio < args.min_ratio:
            # use topr logic.
            max_logits, class_idx = logits.max(dim=-1)
            _, idx = max_logits.topk(dim=-1, k=int(args.min_ratio * logits.size(0)))
            mask_selector = torch.zeros_like(max_logits, dtype=torch.bool)
            mask_selector[idx] = True
        if mask_selector.sum() > 0:
            assert len(mask_selector.size()) == 1 and len(image_ids) == mask_selector.size(0)
            filtered_image_ids = [image_ids[_idx] for _idx in range(len(image_ids)) if mask_selector[_idx]]
            for image_id_field in filtered_image_ids:
                if hasattr(args, "sublist") and args.sublist:
                    example_ids.append(image_id_field)
                else:
                    example_ids.add(image_id_field)

    global ratio
    ratio = len(example_ids) / total_example
    misc.print_json(args.output_dir, json.dumps({"step": step[0], "ratio": ratio, "thres": thres}))
    model.train()
    return example_ids
