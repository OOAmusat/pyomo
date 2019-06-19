#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest

from pyomo.contrib.surrogates.utils import NumpyEvaluator
from pyomo.environ import ConcreteModel, Param, Var, value, sin, atan, atanh
from pyomo.core import ComponentMap

try:
    import numpy as np
    _numpy_available = True
except ImportError:
    _numpy_available = False

@unittest.skipIf(not _numpy_available, "Test requires numpy")
class TestNumpyEvaluator(unittest.TestCase):
    def test_eval_numpy(self):
        m = ConcreteModel()
        m.p = Param([1,2], mutable=True)
        m.x = Var()

        data = np.array([[0,-1,2],
                         [.1,.2,.3],
                         [4,5,6]])
        cMap = ComponentMap()
        cMap[m.p[1]] = data[0]
        cMap[m.p[2]] = data[1]
        cMap[m.x] = data[2]

        npe = NumpyEvaluator(cMap)

        result = npe.walk_expression(sin(m.x))
        self.assertEqual(result[0], sin(4))
        self.assertEqual(result[1], sin(5))
        self.assertEqual(result[2], sin(6))

        result = npe.walk_expression(abs(m.x * m.p[1] - m.p[2]))
        self.assertEqual(result[0], .1)
        self.assertEqual(result[1], -((-1*5)-.2))
        self.assertEqual(result[2], (2*6-.3))

        result = npe.walk_expression(atan(m.x))
        self.assertEqual(result[0], atan(4))
        self.assertEqual(result[1], atan(5))
        self.assertEqual(result[2], atan(6))

        result = npe.walk_expression(atanh(m.p[2]))
        self.assertEqual(result[0], atanh(.1))
        self.assertEqual(result[1], atanh(.2))
        self.assertEqual(result[2], atanh(.3))

    def test_eval_constant(self):
        m = ConcreteModel()
        m.p = Param([1,2], mutable=True)
        m.x = Var(initialize=0.25)

        cMap = ComponentMap()
        cMap[m.p[1]] = 2
        cMap[m.p[2]] = 4

        npe = NumpyEvaluator(cMap)

        expr = m.p[1] + m.p[2] + m.x + .5
        self.assertEqual(npe.walk_expression(expr), 6.75)

        m.p[1] = 2
        m.p[2] = 4
        self.assertEqual(value(expr), 6.75)
