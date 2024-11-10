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

"""
Author: Boris Feld

This module contains logging configuration for Comet

"""
import functools
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import requests

from . import logging_messages
from .constants import DEFAULT_PROJECT_NAME
from .json_encoder import NestedEncoder
from .logging_extensions import formatters
from .logging_extensions.filters.log_filter_setup import setup_filters
from .logging_messages import CONSOLE_DEBUG_DEPRECATED
from .utils import get_user

LOGGER = logging.getLogger(__name__)


def _shorten_record_name(record_name: str) -> Optional[str]:
    """Return the first part of the record (which can be None, comet or comet.connection)"""
    if record_name is not None:
        return record_name.split(".", 1)[0]

    return None


def _map_log_record(record: logging.LogRecord) -> Dict[str, Any]:
    return record.__dict__


class HTTPHandler(logging.Handler):
    def __init__(self, url, api_key, experiment_key):
        super().__init__()
        self.url = url
        self.api_key = api_key
        self.experiment_key = experiment_key
        self.session = requests.Session()

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a record.

        Send the record to the Web server as JSON body
        """
        try:
            payload = {
                "apiKey": self.api_key,
                "record": _map_log_record(record),
                "experimentKey": self.experiment_key,
                "levelname": record.levelname,
                "sender": record.name,
                "shortSender": _shorten_record_name(record.name),
            }
            body = json.dumps(payload, cls=NestedEncoder)

            response = self.session.post(
                self.url,
                data=body,
                headers={"Content-Type": "application/json;charset=utf-8"},
                timeout=10,
            )
            response.raise_for_status()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        super().close()
        self.session.close()

    def handleError(self, record: logging.LogRecord) -> None:
        # Hide errors to avoid bad interaction with console logging
        pass


def expand_log_file_path(log_file_path: str, project_name: Optional[str] = None) -> str:
    """
    Expand patterns in the file logging path.

    Allowed patterns:
        * {datetime}
        * {pid}
        * {project}
        * {user}
    """

    if project_name is None:
        project_name = DEFAULT_PROJECT_NAME

    def make_valid(s):
        # type: (str) -> str
        s = str(s).strip().replace(" ", "_")
        return re.sub(r"(?u)[^-\w.]", "", s)

    user = make_valid(get_user())

    patterns = {
        "datetime": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "pid": os.getpid(),
        "project": project_name,
        "user": user,
    }
    if log_file_path is not None:
        try:
            return log_file_path.format(**patterns)
        except KeyError:
            LOGGER.info(
                "Invalid logging file pattern: '%s'; ignoring",
                log_file_path,
                exc_info=True,
            )
            return log_file_path


class CometLoggingConfig(object):
    def __init__(self):
        self.root: Optional[logging.Logger] = None
        self.console_handler: Optional[logging.StreamHandler] = None
        self.console_formatter: Optional[formatters.CometConsoleFormatter] = None
        self.file_handler: Optional[logging.FileHandler] = None
        self.file_formatter: Optional[formatters.CometFileFormatter] = None

    def setup(self, config) -> logging.Logger:
        self.root = logging.getLogger("comet_ml")
        logger_level = logging.CRITICAL
        message_format = logging_messages.MSG_FORMAT

        file_message_format = logging_messages.FILE_MSG_FORMAT

        # Don't send comet-ml to the application logger
        self.root.propagate = False

        # Add handler for console, basic INFO:
        self.console_handler = logging.StreamHandler()
        logging_console = config["comet.logging.console"]

        if logging_console and logging_console.upper() in [
            "DEBUG",
            "ERROR",
            "INFO",
            "CRITICAL",
            "FATAL",
            "WARN",
            "WARNING",
        ]:
            logging_console_level = logging._checkLevel(logging_console.upper())
            hide_traceback = False
        else:
            logging_console_level = logging.INFO
            hide_traceback = True

        if logging_console_level == logging.DEBUG:
            LOGGER.warning(CONSOLE_DEBUG_DEPRECATED)
            logging_console_level = logging.INFO

        self.console_formatter = formatters.CometConsoleFormatter(
            message_format, hide_traceback=hide_traceback
        )

        self.console_handler.setLevel(logging_console_level)
        self.console_handler.setFormatter(self.console_formatter)
        self.root.addHandler(self.console_handler)

        logger_level = min(logger_level, self.console_handler.level)

        # The std* logger might conflict with the logging if a log record is
        # emitted for each WS message as it would result in an infinite loop. To
        # avoid this issue, all log records after the creation of a message should
        # be at a level lower than info as the console handler is set to info
        # level.

        # Add the additional file handler
        log_file_path = expand_log_file_path(
            config["comet.logging.file"], config["comet.project_name"]
        )
        log_file_level = config["comet.logging.file_level"]
        log_file_overwrite = config["comet.logging.file_overwrite"]

        self.file_handler = None
        self.file_formatter = None

        if log_file_path is not None:

            # Create logfile path, if possible:
            if os.path.dirname(log_file_path) != "":
                try:
                    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                except Exception:
                    LOGGER.error(
                        "can't create path to log file %r", log_file_path, exc_info=True
                    )

            try:
                # Overwrite file if comet.logging.file_overwrite:
                if log_file_overwrite:
                    self.file_handler = logging.FileHandler(log_file_path, "w+")
                else:
                    self.file_handler = logging.FileHandler(log_file_path)

                if log_file_level is not None:
                    log_file_level = logging._checkLevel(log_file_level.upper())
                else:
                    log_file_level = logging.DEBUG

                self.file_handler.setLevel(log_file_level)
                logger_level = min(logger_level, log_file_level)

                if log_file_level == logging.DEBUG:
                    file_message_format = logging_messages.DEBUG_FILE_MSG_FORMAT

                self.file_formatter = formatters.CometFileFormatter(
                    file_message_format, hide_traceback=False
                )
                self.file_handler.setFormatter(self.file_formatter)
                self.root.addHandler(self.file_handler)
            except Exception:
                LOGGER.error(
                    "can't open log file %r; file logging is disabled",
                    log_file_path,
                    exc_info=True,
                )

        self.root.setLevel(logger_level)
        return self.root

    def redact_string(self, string: str) -> None:
        """Blacklists specified string from logging. The string will be masked in the logging."""
        if self.console_formatter:
            self.console_formatter.strings_to_redact.add(string)

        if self.file_formatter:
            self.file_formatter.strings_to_redact.add(string)

    def discard_string_from_redact(self, string: str) -> None:
        """Discards specified string from the set of blacklisted strings. After this method invocation the string will
        no longer be masked from the logger output.
        """
        if self.console_formatter:
            self.console_formatter.strings_to_redact.discard(string)

        if self.file_formatter:
            self.file_formatter.strings_to_redact.discard(string)

    def enable_console_traceback(self) -> None:
        self.console_formatter.enable_traceback()


COMET_LOGGING_CONFIG = CometLoggingConfig()


def _get_comet_logging_config() -> CometLoggingConfig:
    return COMET_LOGGING_CONFIG


def _setup_comet_logging(config):
    logger = COMET_LOGGING_CONFIG.setup(config)
    setup_filters(logger)


def _setup_comet_http_handler(
    url: str, api_key: str, experiment_key: str
) -> HTTPHandler:
    root = logging.getLogger("comet_ml")

    http_handler = HTTPHandler(url, api_key, experiment_key)
    http_handler_level = logging.INFO
    http_handler.setLevel(http_handler_level)

    # Remove any other previous HTTPHandlers:
    root.handlers[:] = [
        handler
        for handler in list(root.handlers)
        if not isinstance(handler, HTTPHandler)
    ]

    # Add new HTTPHandler. Only the current experiment should
    # have its logs forwarded to the backend.
    root.addHandler(http_handler)

    # Ensure the logger level is low enough
    root.setLevel(min(root.level, http_handler_level))

    return http_handler


def convert_exception_to_log_message(
    message: str,
    logger: logging.Logger,
    return_on_exception=None,
    exception_info: bool = True,
    logging_level: int = logging.ERROR,
) -> Callable:
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception:
                logger.log(logging_level, message, exc_info=exception_info)
                return return_on_exception

        return wrapper

    return decorator
