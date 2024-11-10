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
from threading import Event


class FileUploadLimitsGuard:
    def __init__(self) -> None:
        self._image_upload_limit_reached = Event()
        self._embedding_upload_limit_reached = Event()
        self._histogram_upload_limit_reached = Event()

    def image_upload_limit_exceeded(self) -> None:
        self._image_upload_limit_reached.set()

    def has_image_upload_limit_exceeded(self) -> bool:
        return self._image_upload_limit_reached.is_set()

    def embedding_upload_limit_exceeded(self) -> None:
        self._embedding_upload_limit_reached.set()

    def has_embedding_upload_limit_exceeded(self) -> bool:
        return self._embedding_upload_limit_reached.is_set()

    def histogram_upload_limit_exceeded(self) -> None:
        self._histogram_upload_limit_reached.set()

    def has_histogram_upload_limit_exceeded(self) -> bool:
        return self._histogram_upload_limit_reached.is_set()
