# 1. CUTEr problem hs100lnp

from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

m = ConcreteModel()
m.j = Set(initialize=[1, 2, 3, 4, 5, 6, 7])

init_x = {1: 1, 2: 2, 3:0, 4:4, 5: 0, 6: 1, 7: 1}

def x_init(m, i):
    return (init_x[i])

m.x = Var(m.j, initialize = x_init)

def blackbox(a, b, c, d):
   return 127 - 2 * (a ** 2) - 3 * (b ** 4) - 4 * (c ** 2) - 5 * d
bb = ExternalFunction(blackbox)

m.first_constraint = Constraint(expr= m.x[3] == bb(m.x[1], m.x[2], m.x[4], m.x[5]))

m.second_constraint = Constraint(expr= -4 * m.x[1] * m.x[1] - m.x[2] * m.x[2] + 3 * m.x[1] * m.x[2] - 2 * m.x[3] * m.x[3] - 5 * m.x[6] + 11 * m.x[7] == 0)

m.objfunc = Objective(expr=
                     ((m.x[1] - 10)**2 + 5*(m.x[2] - 12)**2 + m.x[3]**4 + 3 * (m.x[4] - 11)**2 + 10 * m.x[5]**6 + 7 * m.x[6]**2 + m.x[7]**4 - 4 * m.x[6] * m.x[7] - 10 * m.x[6] - 8 * m.x[7])
                     )

# m.pprint()

optTRF = SolverFactory('trustregion')
optTRF.solve(m, [bb])

m.display()



# Same problem implemented exactly as in Eason thesis

# from pyomo.environ import *
# from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
#
# m = ConcreteModel()
# m.j = Set(initialize=[1, 2, 3, 4])
# m.z = Var([1, 2], domain=Reals, initialize=1.)
#
# init_x = {1: 1, 2: 2, 3:4, 4: 0}
# def x_init(m, i):
#     return (init_x[i])
# m.x = Var(m.j, initialize=x_init)
#
#
# def blackbox(a, b, c, d):
#    return 127 - 2 * (a ** 2) - 3 * (b ** 4) - 4 * (c ** 2) - 5 * d
# bb = ExternalFunction(blackbox)
#
# q = bb(m.x[1], m.x[2], m.x[3], m.x[4])
#
# m.objfunc = Objective(expr=
#                      ((m.x[1] - 10)**2 + 5*(m.x[2] - 12)**2 + (bb(m.x[1], m.x[2], m.x[3], m.x[4]))**4 + 3 * (m.x[3] - 11)**2 + 10 * m.x[4]**6 + 7 * m.z[1]**2 + m.z[2]**4 - 4 * m.z[1] * m.z[2] - 10 * m.z[1] - 8 * m.z[2])
#                      )
#
# m.second_constraint = Constraint(expr= -4 * m.x[1] * m.x[1] - m.x[2] * m.x[2] + 3 * m.x[1] * m.x[2] - 2 * (bb(m.x[1], m.x[2], m.x[3], m.x[4])) * (bb(m.x[1], m.x[2], m.x[3], m.x[4])) - 5 * m.z[1] + 11 * m.z[2] == 0)
#
#
# # m.pprint()
# optTRF = SolverFactory('trustregion')
# optTRF.solve(m, [bb])
#
# m.display()
