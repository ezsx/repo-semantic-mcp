from __future__ import annotations

import inspect
import unittest


def assert_signature(
    test_case: unittest.TestCase,
    function,
    expected_parameters: list[str],
    expected_defaults: dict[str, object],
) -> None:
    signature = inspect.signature(function)
    test_case.assertEqual(list(signature.parameters), expected_parameters)
    defaulted_parameters = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Signature.empty
    }
    test_case.assertEqual(defaulted_parameters, set(expected_defaults))
    for name, expected in expected_defaults.items():
        test_case.assertEqual(signature.parameters[name].default, expected)
