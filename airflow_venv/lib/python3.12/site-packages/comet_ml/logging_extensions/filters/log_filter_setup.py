# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2024 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import logging

from ...rate_limiter.duration import Duration, TimeUnit
from ...rate_limiter.limiter import OpRateLimiter
from ...rate_limiter.rate import OpRate
from .log_filter import LogRecordRuleFilter
from .log_filter_helpers import type_from_extra_dict
from .log_record_types import (
    EMBEDDING_THROTTLING_MSG_EXTRA,
    HISTOGRAM_THROTTLING_MSG_EXTRA,
    IMAGE_THROTTLING_MSG_EXTRA,
    METRICS_THROTTLING_MSG_EXTRA,
)
from .log_rules import RateLimitedLogRecordRule


def setup_filters(logger: logging.Logger):
    # once in 5 seconds rule
    once_in_5_throttling_rule = RateLimitedLogRecordRule(
        rate_limiter=OpRateLimiter([OpRate(1, Duration(5, TimeUnit.SECOND))]),
        drop_excess_records=True,
    )

    # once in 20 seconds rule
    once_in_20_throttling_rule = RateLimitedLogRecordRule(
        rate_limiter=OpRateLimiter([OpRate(1, Duration(20, TimeUnit.SECOND))]),
        drop_excess_records=True,
    )

    # setup  filter
    log_filter = LogRecordRuleFilter(logger.name)

    # add custom rules for different record types
    log_filter.add_rule(
        once_in_5_throttling_rule,
        record_type=type_from_extra_dict(METRICS_THROTTLING_MSG_EXTRA),
    )

    for record_extra in (
        EMBEDDING_THROTTLING_MSG_EXTRA,
        HISTOGRAM_THROTTLING_MSG_EXTRA,
        IMAGE_THROTTLING_MSG_EXTRA,
    ):
        log_filter.add_rule(
            once_in_20_throttling_rule,
            record_type=type_from_extra_dict(record_extra),
        )

    # add to the handlers
    for handler in logger.handlers:
        handler.addFilter(log_filter)
