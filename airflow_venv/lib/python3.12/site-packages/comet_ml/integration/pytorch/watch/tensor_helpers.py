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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

import torch


def to_numpy(x: torch.Tensor) -> "np.ndarray":
    x = x.detach()  # detach the tensor from computational graph to not corrupt it
    x = x.cpu()  # make sure the tensor located in CPU, not GPU/TPU memory
    x = x.to(
        torch.float32
    )  # make sure we end up with fp32 because pytorch supports different precisions
    x = x.numpy()
    return x
