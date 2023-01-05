# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch

from transformers import VisionTextDualEncoderModel


class CiTCLIPVisionTextDualEncoderModel(VisionTextDualEncoderModel):
    '''a hf model wrapper to support forward with either or both image/text.
    note that HF impl. uses an artificial pooler that most pre-trained models (e.g., ViT) don't have.
    # LiT directly uses [CLS] token for both vision and language.
    text: https://github.com/google-research/vision_transformer/blob/16fc24d2734f34b0a7b16212a4386c41fe662cb4/vit_jax/models_lit.py#L62
    vision: https://github.com/google-research/vision_transformer/blob/16fc24d2734f34b0a7b16212a4386c41fe662cb4/vit_jax/models_vit.py#L283
    configs of LiT: https://github.com/google-research/vision_transformer/blob/16fc24d2734f34b0a7b16212a4386c41fe662cb4/vit_jax/configs/models.py#L319
    '''

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        return_loss=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        skip_text_projection=False,
        split=1,
        **kwargs,
    ):
        image_embeds, text_embeds = None, None
        if pixel_values is not None:
            if split > 1:  # TODO: test if can merge these two branch.
                vision_outputs = []
                for splitted_pixel_values in torch.split(pixel_values, pixel_values.size(0) // split):
                    vision_outputs.append(
                        self.vision_model(
                            pixel_values=splitted_pixel_values,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                        )[1]
                    )
                image_embeds = torch.cat(vision_outputs, dim=0)
            else:
                vision_outputs = self.vision_model(
                    pixel_values=pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                image_embeds = vision_outputs[1]  # pooler_output
            image_embeds = self.visual_projection(image_embeds)

        if input_ids is not None:
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # SimCSE uses pooler as tanh as in HF.
            text_embeds = text_outputs[1]  # pooler_output
            if not skip_text_projection:
                text_embeds = self.text_projection(text_embeds)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        return {"text_embeds": text_embeds, "image_embeds": image_embeds, "logit_scale": logit_scale}


def build_model(args):
    import os
    import hfmodels

    from transformers import AutoTokenizer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"creating model: {args.vision_backbone}-{args.text_backbone}")

    model = CiTCLIPVisionTextDualEncoderModel.from_vision_text_pretrained(  # VisionTextDualEncoderModel
        args.vision_pretrained,  # we dump simclr/moco into HF format.
        args.text_pretrained,  # all text models are in HF. # vision_model= ... your own model is not HF format.
        projection_dim=args.projection_dim if hasattr(args, "projection_dim") else 512
    )
    tokenizer = AutoTokenizer.from_pretrained(args.text_pretrained, use_fast=True)
    return model, tokenizer
