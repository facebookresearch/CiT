# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch


from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

import timm
assert timm.__version__ >= "0.4.12", "make sure timm uses augreg checkpoints."


class AugRegConfig(PretrainedConfig):
    """
    HF or older timm doesn't load augreg weights.
    """
    model_type = "augreg"

    def __init__(
        self,
        config_name="vit_base_patch32_224_in21k",
        hidden_size=768,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config_name = config_name
        self.hidden_size = hidden_size


AutoConfig.register("augreg", AugRegConfig)


class AugRegModel(PreTrainedModel):
    config_class = AugRegConfig

    @classmethod
    def from_orig_pretrained(cls, config_name):
        augreg = timm.create_model(config_name, pretrained=True)
        config = AugRegConfig(config_name=config_name, hidden_size=augreg.embed_dim)
        model = AugRegModel(config)
        model.augreg = augreg
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        import os
        ckpt_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        state_dict = torch.load(os.path.join(ckpt_path))
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = AugRegModel(config)
        model.load_state_dict(state_dict, strict=True)
        return model

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.augreg = timm.create_model(config.config_name, pretrained=False)
        self.post_init()

    def _init_weights(self, module):
        self.augreg._init_weights(module)

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
        # https://github.com/rwightman/pytorch-image-models/blob/e0c4eec4b66dc14ae96097c7b4a7ef2af45ba309/timm/models/vision_transformer.py#L358
        # pre_logits is nn.Identity and token means from CLS [:, 0]
        sequence_output = self.augreg.forward_features(pixel_values)
        pooled_output = sequence_output

        if not return_dict:
            return (sequence_output, pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=None,  # encoder_outputs.hidden_states,
            attentions=None,  # encoder_outputs.attentions,
        )


AutoModel.register(AugRegConfig, AugRegModel)


if __name__ == '__main__':
    # dump this model for AutoModel: `python -m hfmodels.augreg`
    models = ["vit_base_patch32_224_in21k", "vit_base_patch16_224_in21k", "vit_large_patch16_224_in21k"]
    for model in models:
        vision_model = AugRegModel.from_orig_pretrained(model)
        vision_model.save_pretrained(f"pretrained_models/{model}_augreg_hf")
