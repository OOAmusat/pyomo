#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr import current as EXPR, native_types
from pyomo.core.expr.numvalue import value
from pyomo.core.kernel.component_map import ComponentMap

_numpy_available = True
try:
    import numpy

    _functionMap = {
        'exp': numpy.exp,
        'log': numpy.log,
        'log10': numpy.log10,
        'sin': numpy.sin,
        'asin': numpy.arcsin,
        'sinh': numpy.sinh,
        'asinh': numpy.arcsinh,
        'cos': numpy.cos,
        'acos': numpy.arccos,
        'cosh': numpy.cosh,
        'acosh': numpy.arccosh,
        'tan': numpy.tan,
        'atan': numpy.arctan,
        'tanh': numpy.tanh,
        'atanh': numpy.arctanh,
        'ceil': numpy.ceil,
        'floor': numpy.floor,
        'sqrt': numpy.sqrt,
    }
except ImportError:
    _numpy_available = False


class NumpyEvaluator(EXPR.StreamBasedExpressionVisitor):

    def __init__(self, object_map):
        super(NumpyEvaluator, self).__init__()
        self.object_map = object_map

    def exitNode(self, node, values):
        if node.__class__ is EXPR.UnaryFunctionExpression or \
           node.__class__ is EXPR.NPV_UnaryFunctionExpression:
            return _functionMap[node._name](values[0])
        if node.__class__ is EXPR.AbsExpression or \
           node.__class__ is EXPR.NPV_AbsExpression:
            return numpy.abs(values[0])
        return node._apply_operation(values)

    def beforeChild(self, node, child):
        #
        # Don't replace native types
        #
        if type(child) in native_types:
            return False, child
        #
        # We will descend into all expressions...
        #
        if child.is_expression_type():
            return True, None
        #
        # Replace pyomo variables with numpy variables
        #
        if child in self.object_map:
            return False, self.object_map[child]
        #
        # Assume everything else is a constant...
        #
        return False, value(child)