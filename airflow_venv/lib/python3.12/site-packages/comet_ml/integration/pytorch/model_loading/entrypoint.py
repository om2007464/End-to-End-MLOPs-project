# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2021 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

import logging
from typing import Any, Optional

import comet_ml
from comet_ml import _reporting, connection
from comet_ml.model_downloader.uri import parse, scheme

from .. import model_metadata
from ..types import ModelStateDict, Module
from . import load

LOGGER = logging.getLogger(__name__)


def load_model(
    model_uri: str,
    map_location: Any = None,
    pickle_module: Optional[Module] = None,
    **torch_load_args
) -> ModelStateDict:
    """
    Load model's state_dict from experiment, registry or from disk by uri. This will returns a
    Pytorch state_dict that you will need to load into your model. This will load the model using
    [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html).

    Args:
        model_uri: string (required), a uri string defining model location. Possible options are:

            - file://data/my-model
            - file:///path/to/my-model
            - registry://workspace/registry_name (takes the last version)
            - registry://workspace/registry_name:version
            - experiment://experiment_key/model_name
            - experiment://workspace/project_name/experiment_name/model_name
        map_location: Passed to torch.load (see [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html))
        pickle_module: Passed to torch.load (see [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html))
        torch_load_args: Passed to torch.load (see [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html))

    Returns: model's state dict

    Example:
        Here is an example of loading a model from the Model Registry for inference:

        ```python
        from comet_ml.integration.pytorch import load_model

        class TheModelClass(nn.Module):
            def __init__(self):
                super(TheModelClass, self).__init__()
                ...

            def forward(self, x):
                ...
                return x

        # Initialize model
        model = TheModelClass()

        # Load the model state dict from Comet Registry
        model.load_state_dict(load_model("registry://WORKSPACE/TheModel:1.2.4"))

        model.eval()

        prediction = model(...)
        ```

        Here is an example of loading a model from an Experiment for Resume Training:

        ```python
        from comet_ml.integration.pytorch import load_model

        # Initialize model
        model = TheModelClass()

        # Load the model state dict from a Comet Experiment
        checkpoint = load_model("experiment://e1098c4e1e764ff89881b868e4c70f5/TheModel")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model.train()
        ```
    """
    if pickle_module is None:
        pickle_module = model_metadata.get_torch_pickle_module()

    if parse.request_type(model_uri) == parse.RequestTypes.UNDEFINED:
        raise ValueError("Invalid model_uri: '{}'".format(model_uri))

    if scheme.is_file(model_uri):
        model = load.from_disk(
            model_uri,
            map_location=map_location,
            pickle_module=pickle_module,
            **torch_load_args
        )
    else:
        model = load.from_remote(
            model_uri,
            map_location=map_location,
            pickle_module=pickle_module,
            **torch_load_args
        )
        _load_model_track_usage(model_uri)

    return model


def _load_model_track_usage(model_uri: str) -> None:
    config = comet_ml.get_config()

    connection.Reporting.report(
        config=config,
        api_key=comet_ml.get_api_key(None, config),
        event_name=_reporting.PYTORCH_MODEL_LOADING_EXPLICIT_CALL,
        err_msg=model_uri,
    )
