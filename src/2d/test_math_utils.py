# test_math_utils.py


import unittest
import numpy as np
import casadi as ca
import math_utils


class TestMathUtils(unittest.TestCase):
    
    def test_numpy_matrix(self):
        mat = np.zeros((3, 4))
        self.assertEqual(math_utils.getMRow(mat), 3)
        self.assertEqual(math_utils.getMColumn(mat), 4)

    def test_casadi_SX_matrix(self):
        mat = ca.SX.zeros(5, 2)
        self.assertEqual(math_utils.getMRow(mat), 5)
        self.assertEqual(math_utils.getMColumn(mat), 2)

    def test_casadi_MX_matrix(self):
        mat = ca.MX.zeros(7, 1)
        self.assertEqual(math_utils.getMRow(mat), 7)
        self.assertEqual(math_utils.getMColumn(mat), 1)

    def test_unsupported_type(self):
        with self.assertRaises(TypeError):
            math_utils.getMRow("bad input")


if __name__ == '__main__':
    from test_runner import TestRunner
    unittest.main(testRunner=TestRunner(), verbosity=2)