from __future__ import annotations

import numpy as np
import pytest

from relate import PYTHON_RELATION_NAMES, extract_python_structure


def test_extracts_original_three_python_relations() -> None:
    structure = extract_python_structure(
        """
def process(value):
    if value and helper(value):
        return client.send(value)
    return helper(value)
"""
    )

    assert PYTHON_RELATION_NAMES == (
        "cyclomatic_complexity",
        "max_control_depth",
        "distinct_call_sites",
    )
    np.testing.assert_array_equal(structure.as_array(), np.asarray([3.0, 1.0, 2.0]))


def test_nested_functions_do_not_leak_into_outer_structure() -> None:
    structure = extract_python_structure(
        """
def outer(value):
    def nested():
        if value:
            nested_call()
    return outer_call(value)
"""
    )

    np.testing.assert_array_equal(structure.as_array(), np.asarray([1.0, 0.0, 1.0]))


def test_requires_one_top_level_function() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        extract_python_structure("x = 1")
