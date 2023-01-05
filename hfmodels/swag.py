# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch

from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModel,
)

from transformers.modeling_outputs import BaseModelOutputWithPooling


class SwagConfig(PretrainedConfig):
    model_type = "swag"

    def __init__(
        self,
        config_name="vit_b16",
        hidden_size=768,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config_name = config_name
        self.hidden_size = hidden_size


AutoConfig.register("swag", SwagConfig)


class SwagModel(PreTrainedModel):
    config_class = SwagConfig

    @classmethod
    def from_orig_pretrained(cls, config_name):
        swag = torch.hub.load("facebookresearch/swag", model=config_name)
        config = SwagConfig(config_name=config_name, hidden_size=swag.hidden_dim)
        model = SwagModel(config)
        model.swag = swag
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        import os
        ckpt_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        state_dict = torch.load(os.path.join(ckpt_path))
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = SwagModel(config)
        model.load_state_dict(state_dict, strict=True)
        return model

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.swag = torch.hub.load("facebookresearch/swag", model=config.config_name)
        self.post_init()

    def _init_weights(self, module):
        self.swag.init_weights()  # check existence.

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
        sequence_output = self.swag(pixel_values)
        pooled_output = sequence_output

        if not return_dict:
            return (sequence_output, pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=None,  # encoder_outputs.hidden_states,
            attentions=None,  # encoder_outputs.attentions,
        )


AutoModel.register(SwagConfig, SwagModel)


if __name__ == '__main__':
    # dump this model for AutoModel: `python -m hfmodels.swag`
    models = ["vit_b16", "vit_l16", "vit_h14"]
    for model in models:
        vision_model = SwagModel.from_orig_pretrained(model)
        vision_model.save_pretrained(f"pretrained_models/{model}_swag_hf")