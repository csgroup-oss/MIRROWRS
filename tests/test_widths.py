# Copyright (C) 2024-2025 CS GROUP, https://csgroup.eu
# Copyright (C) 2024 CNES.
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
module test_widths.py
: Unit tests for module mirrowrs/widths.py
"""

import pytest

from mirrowrs.widths import ParamWidthComp

@pytest.fixture
def dct_config_kwargs():
    """Return a kwargs-like dictionary to use with the ParamWidthComp class
    :return dct_out: dict
    """

    dct_out = {
        "label_attr" : "label",
        "min_width": 50.,
        "export_buffered_sections": False,
        "bool_print_dry": True
    }

    return dct_out

# Test ParamWidthComp instantiation : with default values
def test_paramwidthcomp_init_default():
    """Test ParamWidthComp instantiation : with default values
    """

    obj = ParamWidthComp
    assert obj.label_attr is None
    assert obj.min_width == -1
    assert obj.export_buffered_sections is False
    assert obj.bool_print_dry is False

# Test ParamWidthComp instantiation : with specified values
def test_paramwidthcomp_init(dct_config_kwargs):
    """Test ParamWidthComp instantiation : with specified values
    """

    obj = ParamWidthComp(**dct_config_kwargs)
    assert obj.label_attr == "label"
    assert obj.min_width == 50.
    assert obj.export_buffered_sections is False
    assert obj.bool_print_dry is True

# Test ParamWidthComp method __post_init__()
@pytest.mark.parametrize("key, wrong_value",
                         [("label_attr", 1),
                          ("min_width", "a"),
                          ("export_buffered_sections", "a"),
                          ("bool_print_dry", "a")],)
def test_paramwidthcomp_wrong_input(key, wrong_value, dct_config_kwargs):
    """Test method ParamWidthComp.__post_init__()
    """
    dct_test = dct_config_kwargs.copy()
    dct_test[key] = wrong_value
    with pytest.raises(TypeError):
        _ = ParamWidthComp(**dct_test)