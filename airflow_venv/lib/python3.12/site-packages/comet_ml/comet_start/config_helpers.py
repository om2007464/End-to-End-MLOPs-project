# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2023 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
from typing import Any, Dict

import comet_ml.api


def update_with_default(
    key: str, default: Any, values_dict: Dict[str, Any]
) -> Dict[str, Any]:
    if values_dict.get(key) is None:
        values_dict[key] = default

    return values_dict


def check_experiment_already_exists(
    api_key: str,
    experiment_key: str,
) -> bool:
    api = comet_ml.api.get_instance(api_key=api_key, cache=False)

    experiment = api.get_experiment_by_key(experiment_key=experiment_key)
    return experiment is not None
