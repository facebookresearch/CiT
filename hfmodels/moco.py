# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch

import sys
sys.path.append("moco-v3")  # repo path to moco-v3

from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModel,
)

from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from vits import vit_base
from functools import partial
from moco.builder import MoCo_ViT
from collections import OrderedDict


class MoCoConfig(PretrainedConfig):
    """
    refer `https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/vit/configuration_vit.py#L29`
    `model_type` only has three choices.
    https://github.com/huggingface/transformers/blob/05fa1a7ac17bb7aa07b9e0c1e138ecb31a28bbfe/src/transformers/models/vision_text_dual_encoder/configuration_vision_text_dual_encoder.py#L94
    how to make sure `hidden_size` match checkpoint ?
    """
    model_type = "moco"

    def __init__(
        self,
        config_name="vit_base_patch16",
        hidden_size=256,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config_name = config_name
        self.hidden_size = hidden_size


AutoConfig.register("moco", MoCoConfig)


class MoCoModel(PreTrainedModel):
    config_class = MoCoConfig

    @classmethod
    def from_orig_pretrained(cls, ckpt_dir):
        """load from original checkpoint; used to save a HF checkpoint, see main."""
        config = MoCoConfig(hidden_size=256)
        model = MoCoModel(config)
        print("loading weights from", ckpt_dir)
        ckpt = torch.load(ckpt_dir, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            k = k.replace('module.', '')
            for prefix in ["momentum_encoder", "predictor"]:
                if k.startswith(prefix):
                    break
            else:
                state_dict[k.replace("base_encoder.", "")] = v
        model.moco.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        import os
        ckpt_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        state_dict = torch.load(os.path.join(ckpt_path))
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = MoCoModel(config)
        model.load_state_dict(state_dict, strict=True)
        return model

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.moco = MoCo_ViT(
            partial(vit_base, stop_grad_conv1=True),
            256, 4096, 0.2
        ).base_encoder
        self.post_init()

    def _init_weights(self, m):
        # borrowed from mae
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        pixel_values=None,
        # attention_mask=None,
        # head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        # interpolate_pos_encoding=None,
        return_dict=None
    ):
        encoder_outputs = self.moco(pixel_values)
        sequence_output = encoder_outputs.unsqueeze(1)
        pooled_output = encoder_outputs
        if not return_dict:
            return (sequence_output, pooled_output) # + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=encoder_outputs,
            hidden_states=None,  # encoder_outputs.hidden_states,
            attentions=None,  # encoder_outputs.attentions,
        )


AutoModel.register(MoCoConfig, MoCoModel)


if __name__ == '__main__':
    # dump this model for AutoModel: `python -m hfmodels.moco`
    vision_model = MoCoModel.from_orig_pretrained("pretrained_models/moco/vit-b-300ep.pth.tar")
    vision_model.save_pretrained("pretrained_models/moco_hf")
