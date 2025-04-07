# Copyright (C) 2024-2025 CS GROUP, https://csgroup.eu
#
# This file is part of MIRROWRS (Earth Observations For HydrauDynamic Model Generation)
#
#     https://github.com/csgroup-oss/MIRROWRS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
tools.py
: module containing diverse python tools to follow execution
"""

import logging
import time

_logger = logging.getLogger("tools_module")


class FileExtensionError(TypeError):
    """Specific exception indicating a wrong file extension
    """
    def __init__(self, message="File has to have a different extension file"):
        """Class constructor"""
        self.message = message
        super().__init__(self.message)


class DisjointBboxError(ValueError):
    """Specific exception indicating that two bounding boxes are disjoint
    """
    def __init__(self, message="Compared scenes do not overlap."):
        """Class constructor"""
        self.message = message
        super().__init__(self.message)


class DimensionError(ValueError):
    """Specific exception indicating the object does not have the right shape
    """
    def __init__(self, message="Object has the wrong dimensions."):
        """Class constructor"""
        self.message = message
        super().__init__(self.message)


class Timer:
    """Class dedicating to time a run
    """
    def __init__(self):
        """Class constructor"""

        self.start_time = 0.0
        self.tmp_time = 0.0
        self.stop_time = 0.0

    def start(self):
        """Initialize the time"""

        self.start_time = time.time()

    def stop(self):
        """
        :return message: str
            To display total duration time in seconds
        """

        # Calculate time since Timer initialization
        self.stop_time = time.time() - self.start_time

        # Output message
        message = "Total execution time in %s" % self.stop_time

        return message

    def info(self, flag_start=1):
        """Current time

        :param flag_start: int
            set whether temporary duration is from the Timer initialiation (=1) or not
        :return message: str
             To display total temporary time in seconds
        """

        # calculate duration time
        if (self.tmp_time == 0) or (flag_start == 1):
            current_time = time.time() - self.start_time
        else:
            current_time = time.time() - self.tmp_time

        # Update temporary time
        self.tmp_time = time.time()

        # Output message
        message = "Step executed in %s" % current_time

        return message
