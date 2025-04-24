# Copyright (C) 2024 CNES.
#
# This file is part of MIRROWRS (Mapper to InfeR River Observations of Widths from Remote Sensing)
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
module test_tools.py
: Unit tests for module mirrowrs/tools.py
"""

import pytest

from mirrowrs.tools import FileExtensionError
from mirrowrs.tools import DisjointBboxError
from mirrowrs.tools import DimensionError

# Test if it is a subclass of TypeError
def test_fileextentionerror_is_subclass():
    assert issubclass(FileExtensionError, TypeError)

# Test if exception can be raised correctly
def test_fileextentionerror_can_be_raised_and_caught():
    with pytest.raises(FileExtensionError) as exc_info:
        raise FileExtensionError
    assert isinstance(exc_info.value, TypeError)

# Test error message
def test_fileextentionerror_error_message():
    err = FileExtensionError("Oops!")
    assert str(err) == "Oops!"

# Test if it is a subclass of ValueError
def test_disjointbboxerror_is_subclass():
    assert issubclass(DisjointBboxError, ValueError)

# Test if exception can be raised correctly
def test_disjointbboxerror_can_be_raised_and_caught():
    with pytest.raises(DisjointBboxError) as exc_info:
        raise DisjointBboxError
    assert isinstance(exc_info.value, ValueError)

# Test error message
def test_disjointbboxerror_error_message():
    err = DisjointBboxError("Oops!")
    assert str(err) == "Oops!"

# Test if it is a subclass of ValueError
def test_dimensionerror_is_subclass():
    assert issubclass(DimensionError, ValueError)

# Test if exception can be raised correctly
def test_dimensionerror_can_be_raised_and_caught():
    with pytest.raises(DimensionError) as exc_info:
        raise DimensionError
    assert isinstance(exc_info.value, ValueError)

# Test error message
def test_dimensionerror_error_message():
    err = DimensionError("Oops!")
    assert str(err) == "Oops!"