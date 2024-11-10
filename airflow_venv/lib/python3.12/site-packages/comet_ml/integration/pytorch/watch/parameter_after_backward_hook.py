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

import comet_ml
import comet_ml._logging

import torch

from . import tensor_helpers

LOGGER = logging.getLogger(__name__)


class ParameterAfterBackwardHook:
    def __init__(
        self,
        experiment: comet_ml.BaseExperiment,
        name: str,
        parameter: torch.nn.Parameter,
        log_step_interval: int,
        initial_step: int,
    ) -> None:
        self._experiment = experiment

        self._name = name
        self._log_step_interval = log_step_interval
        self._current_step = initial_step
        self._parameter = parameter

        self._gradients = 0  # it will become torch tensor after the first accumulation

    def __call__(self, grad: torch.Tensor) -> None:
        self._current_step += 1
        self._gradients += grad
        needs_logging = (self._current_step % self._log_step_interval) == 0

        if not needs_logging:
            return

        try:
            self._experiment.log_histogram_3d(
                tensor_helpers.to_numpy(self._gradients).flatten(),
                "gradient/{}".format(self._name),
                step=self._current_step,
            )

            self._experiment.log_histogram_3d(
                tensor_helpers.to_numpy(self._parameter).flatten(),
                self._name,
                step=self._current_step,
            )
        except Exception:
            LOGGER.debug(
                "Failed to log model weights and gradients at step %d",
                self._current_step,
                exc_info=True,
            )

        self._gradients = 0
