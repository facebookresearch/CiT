# Copyright (c) Meta Platforms, Inc. All Rights Reserved

"""
pre-configed trainable weights.
"""


pre_projection_weights = ['logit_scale', 'visual_projection.weight', 'text_projection.weight']

# TODO: unify layer selection for all models.
pre_vision_trainable_weights = {
    "moco": {
        "head": ['moco.head'],
        "none": [], 
        "all": ["[ALL]"]
    },
    "augreg": {
        "none": [],
        "all": ["[ALL]"],
    },
    "swag": {
        "none": [],
        "all": ["[ALL]"],
    }
}

pre_text_trainable_weights = {
    "bert": {
        "pool": ['pooler.dense.weight', 'pooler.dense.bias'],
        "all": ["[ALL]"]
    },
}


def _freeze_model(model, trainable_weights):
    '''we assume pretrained model has unknown freezing status.
    all model must pass through this function.
    (e.g.,, MoCo teacher is freezed after pretraining.
    [ALL] indicates fully trainable.
    '''
    for name, parameter in model.named_parameters():
        for param in trainable_weights:
            if name.startswith(param) or param == "[ALL]":
                parameter.requires_grad = True
                break
        else:
            parameter.requires_grad = False

    trainable_parameters = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_parameters.append(name)
    print(f"{model.__class__.__name__} trainable weights:", trainable_parameters)


def freeze_model(model, args):
    assert "-" in args.trainable_weight, "trainable_weight needs format <vision_weight_config>-<text_weight_config>."
    vision_config, text_config = args.trainable_weight.split("-")
    vision_trainable_weights = pre_vision_trainable_weights[args.vision_backbone][vision_config]
    text_trainable_weights = pre_text_trainable_weights[args.text_backbone][text_config]
    _freeze_model(model, pre_projection_weights)
    _freeze_model(model.vision_model, vision_trainable_weights)
    _freeze_model(model.text_model, text_trainable_weights)
    return model
