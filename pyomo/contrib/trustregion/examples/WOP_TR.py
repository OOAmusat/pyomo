# All three rate equations

from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyutilib.math import infinity

model = ConcreteModel()

model.i = Set(initialize=['a', 'b', 'c', 'e', 'p', 'g'])

model.rho = Param(initialize=50)

init_x = {'a': 0.193, 'b': 0.576, 'c': 0.0577, 'e': 0.0577, 'p': 0.1, 'g': 0.0192}
def x_init(model, i):
    return (init_x[i])
model.x = Var(model.i,  initialize=x_init, bounds=(0, 1))


model.F_eff = Var(model.i, initialize=0, bounds=(0, infinity))
model.F_eff['a'] = 10.0
model.F_eff['b'] = 30.0
model.F_eff['c'] = 3.0
model.F_eff['e'] = 3.0
model.F_eff['p'] = 5.0


lb = {'a': 1, 'b': 1, 'c': 0, 'e': 0, 'p': 0, 'g': 0}
ub = {'a': infinity, 'b': infinity, 'c': infinity, 'e': infinity, 'p': 4.763, 'g': infinity}

def f_bounds(model, i):
    return (lb[i], ub[i])


model.F = Var(model.i, initialize=0, bounds=f_bounds)
model.F['a'] = 10.0
model.F['b'] = 20.0
model.F['g'] = 1.0
model.F['p'] = 0.5

model.F_r = Var(model.i, initialize=0, bounds=(0, infinity))
model.F_sum_eff = Var(initialize=52, bounds=(1, infinity))
model.F_purge = Var(initialize=0.0, bounds=(0, infinity))

model.n_eff = Var(initialize=0., bounds=(0, 1))
model.V = Var(initialize=0.06, bounds=(0.03, 0.1))
model.T = Var(initialize=5.80, bounds=(5.8, 6.8))

# ********************** External function section***********************************************
a1 = 5.9755e9
a2 = 2.5962e12
a3 = 9.6283e15
rho = 50


def blackbox_1(comp_1, comp_2, temp, vol):
    return a1 * comp_1 * comp_2 * vol * rho * exp(-120 / temp)
bb_1 = ExternalFunction(blackbox_1)


def blackbox_2(comp_1, comp_2, temp, vol):
    return a2 * comp_1 * comp_2 * vol * rho * exp(-150 / temp)
bb_2 = ExternalFunction(blackbox_2)


def blackbox_3(comp_1, comp_2, temp, vol):
    return a3 * comp_1 * comp_2 * vol * rho * exp(-200 / temp)
bb_3 = ExternalFunction(blackbox_3)

model.r1 = Var(initialize=2.1, bounds=(0, infinity))
model.r2 = Var(initialize=1.5, bounds=(0, infinity))
model.r3 = Var(initialize=0.2, bounds=(0, infinity))
model.rex_c1 = Constraint(expr= model.r1 == bb_1(model.x['a'], model.x['b'], model.T, model.V))
model.rex_c2 = Constraint(expr= model.r2 == bb_2(model.x['b'], model.x['c'], model.T, model.V))
model.rex_c3 = Constraint(expr= model.r3 == bb_3(model.x['p'], model.x['c'], model.T, model.V))

# ***********************************************************************************************

model.rbe_c1 = Constraint(expr=model.F_eff['a'] - model.F['a'] - model.F_r['a'] + model.r1 == 0)

model.rbe_c2 = Constraint(expr=model.F_eff['b'] - model.F['b'] - model.F_r['b'] + model.r1 + model.r2 == 0)

model.rbe_c3 = Constraint(expr=model.F_eff['c'] - model.F_r['c'] - 2 * model.r1 + model.r2 * 2 + model.r3 == 0)

model.rbe_c4 = Constraint(expr=model.F_eff['e'] - model.F_r['e'] - model.r2 * 2  == 0)

model.rbe_c5 = Constraint(expr=model.F_eff['p'] - 0.1 * model.F_r['e'] - model.r2 + model.r3 * 0.5 == 0)

model.rbe_c6 = Constraint(expr=model.F_eff['g'] == model.r3 * 1.5)

model.rbe_c7 = Constraint(expr=model.F_sum_eff - sum(model.F_eff[m] for m in model.i) == 0)

def mole_frac_eval(model, i):
    return model.F_eff[i] == model.F_sum_eff * model.x[i]

model.rbe_c8 = Constraint(model.i, rule=mole_frac_eval, doc='Define mole fractions')

model.rbe_c9 = Constraint(expr= -(model.F_sum_eff - 1) <= 0)

model.ws_c1 = Constraint(expr=model.F['g'] - model.F_eff['g'] == 0)

model.ps_c1 = Constraint(expr=model.F['p'] == model.F_eff['p'] - 0.1 * model.F_eff['e'])

model.pg_c1 = Constraint(expr=model.F_purge == (model.F_eff['a'] + model.F_eff['b'] + model.F_eff['c'] + 1.1 * model.F_eff['e']) * model.n_eff)

def recycle_eval(model, i):
    return model.F_r[i] == model.F_eff[i] * (1 - model.n_eff)

model.rs_c1 = Constraint(model.i, rule=recycle_eval, doc='Define recycle stream')


model.ROI = Objective(
    expr= -((2207 * model.F['p'] + 50 * model.F_purge - 168 * model.F['a'] - 252 * model.F['b'] - 2.22 * model.F_sum_eff - 84 * model.F['g'] - 60 * model.V * model.rho) / (6 * model.V *  model.rho)), sense=minimize
                      )
model.pprint()
instance = model
optTRF = SolverFactory('trustregion')
result = optTRF.solve(instance, [bb_1, bb_2, bb_3])
instance.display()


# # Version with only one reaction as black box
# from pyomo.environ import *
# from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
# from pyutilib.math import infinity
#
# model = ConcreteModel()
#
# model.i = Set(initialize=['a', 'b', 'c', 'e', 'p', 'g'])
#
# model.rho = Param(initialize=50)
#
# init_x = {'a': 0.193, 'b': 0.576, 'c': 0.0577, 'e': 0.0577, 'p': 0.1, 'g': 0.0192}
# def x_init(model, i):
#     return (init_x[i])
# model.x = Var(model.i,  initialize=x_init, bounds=(0, 1))
#
#
# model.F_eff = Var(model.i, initialize=0, bounds=(0, infinity))
# model.F_eff['a'] = 10.0
# model.F_eff['b'] = 30.0
# model.F_eff['c'] = 3.0
# model.F_eff['e'] = 3.0
# model.F_eff['p'] = 5.0
#
#
# lb = {'a': 1, 'b': 1, 'c': 0, 'e': 0, 'p': 0, 'g': 0}
# ub = {'a': infinity, 'b': infinity, 'c': infinity, 'e': infinity, 'p': 4.763, 'g': infinity}
#
# def f_bounds(model, i):
#     return (lb[i], ub[i])
#
#
# model.F = Var(model.i, initialize=0, bounds=f_bounds)
# model.F['a'] = 10.0
# model.F['b'] = 20.0
# model.F['g'] = 1.0
# model.F['p'] = 0.5
#
# model.F_r = Var(model.i, initialize=0, bounds=(0, infinity))
# model.F_sum_eff = Var(initialize=52, bounds=(1, infinity))
# model.F_purge = Var(initialize=0.0, bounds=(0, infinity))
#
# model.n_eff = Var(initialize=0., bounds=(0, 1))
# model.V = Var(initialize=0.06, bounds=(0.03, 0.1))
# model.T = Var(initialize=5.80, bounds=(5.8, 6.8))
#
# # ********************** External function section***********************************************
# a3 = 9.6283e15
# rho = 50
#
# def blackbox_3(comp_1, comp_2, temp, vol):
#     return a3 * comp_1 * comp_2 * vol * rho * exp(-200 / temp)
# bb_3 = ExternalFunction(blackbox_3)
# model.r3 = Var(initialize=0, bounds=(0, infinity))
# model.rex_c3 = Constraint(expr= model.r3 == bb_3(model.x['p'], model.x['c'], model.T, model.V))
#
# model.r1 = Var(initialize=3)
# model.r2 = Var(initialize=2)
# model.T2 = Var(initialize = 0.147, bounds=(0.147, 1.725))
# model.j = Set(initialize=[1, 2])
#
# model.q = Param(model.j, mutable=True)
# model.q[1] = 120
# model.q[2] = 150
#
# model.a = Param(model.j, mutable=True)
# model.a[1] = 5.9755e9
# model.a[2] = 2.5962e12
#
# model.ac = Var(model.j, initialize=0)
#
# model.kf = Var(model.j)
# model.kf[1] = 6.18
# model.kf[2] = 15.2
#
#
# def k_constraint(model, j):
#     return model.ac[j] == log(model.a[j]) - model.q[j] * model.T2
#
# model.re1 = Constraint(model.j, rule=k_constraint, doc='K-prime evaluation')
#
# model.re2 = Constraint(expr=model.T * model.T2 == 1)
#
# def kf_constraint(model, j):
#     return model.kf[j] == exp(model.ac[j])
#
# model.re3 = Constraint(model.j, rule=kf_constraint, doc='exponential term')
#
# def reaction_rate_1_expression(model):
#     return model.r1 == model.kf[1] * model.x['a'] * model.x['b'] * model.V * model.rho
#
# model.re4 = Constraint(rule=reaction_rate_1_expression, doc='r1 expression')
#
# def reaction_rate_2_expression(model):
#     return model.r2 == model.kf[2] * model.x['c'] * model.x['b'] * model.V * model.rho
#
# model.re5 = Constraint(rule=reaction_rate_2_expression, doc='r2 expression')
# # ***********************************************************************************************
#
# model.rbe_c1 = Constraint(expr=model.F_eff['a'] - model.F['a'] - model.F_r['a'] + model.r1 == 0)
#
# model.rbe_c2 = Constraint(expr=model.F_eff['b'] - model.F['b'] - model.F_r['b'] + model.r1 + model.r2 == 0)
#
# model.rbe_c3 = Constraint(expr=model.F_eff['c'] - model.F_r['c'] - 2 * model.r1 + model.r2 * 2 + model.r3 == 0)
#
# model.rbe_c4 = Constraint(expr=model.F_eff['e'] - model.F_r['e'] - model.r2 * 2  == 0)
#
# model.rbe_c5 = Constraint(expr=model.F_eff['p'] - 0.1 * model.F_r['e'] - model.r2 + model.r3 * 0.5 == 0)
#
# model.rbe_c6 = Constraint(expr=model.F_eff['g'] == model.r3 * 1.5)
#
# model.rbe_c7 = Constraint(expr=model.F_sum_eff - sum(model.F_eff[m] for m in model.i) == 0)
#
# def mole_frac_eval(model, i):
#     return model.F_eff[i] == model.F_sum_eff * model.x[i]
#
# model.rbe_c8 = Constraint(model.i, rule=mole_frac_eval, doc='Define mole fractions')
#
# model.rbe_c9 = Constraint(expr= -(model.F_sum_eff - 1) <= 0)
#
# model.ws_c1 = Constraint(expr=model.F['g'] - model.F_eff['g'] == 0)
#
# model.ps_c1 = Constraint(expr=model.F['p'] == model.F_eff['p'] - 0.1 * model.F_eff['e'])
#
# model.pg_c1 = Constraint(expr=model.F_purge == (model.F_eff['a'] + model.F_eff['b'] + model.F_eff['c'] + 1.1 * model.F_eff['e']) * model.n_eff)
#
# def recycle_eval(model, i):
#     return model.F_r[i] == model.F_eff[i] * (1 - model.n_eff)
#
# model.rs_c1 = Constraint(model.i, rule=recycle_eval, doc='Define recycle stream')
#
#
# model.ROI = Objective(
#     expr= -((2207 * model.F['p'] + 50 * model.F_purge - 168 * model.F['a'] - 252 * model.F['b'] - 2.22 * model.F_sum_eff - 84 * model.F['g'] - 60 * model.V * model.rho) / (6 * model.V *  model.rho)), sense=minimize
#                       )
#
# instance = model
# optTRF = SolverFactory('trustregion')
# result = optTRF.solve(instance, [bb_3])
# instance.display()





# # Two reactions
#
# from pyomo.environ import *
# from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
# from pyutilib.math import infinity
#
# model = ConcreteModel()
#
# model.i = Set(initialize=['a', 'b', 'c', 'e', 'p', 'g'])
#
# model.rho = Param(initialize=50)
#
# init_x = {'a': 0.193, 'b': 0.576, 'c': 0.0577, 'e': 0.0577, 'p': 0.1, 'g': 0.0192}
# def x_init(model, i):
#     return (init_x[i])
# model.x = Var(model.i,  initialize=x_init, bounds=(0, 1))
#
#
# model.F_eff = Var(model.i, initialize=0, bounds=(0, infinity))
# model.F_eff['a'] = 10.0
# model.F_eff['b'] = 30.0
# model.F_eff['c'] = 3.0
# model.F_eff['e'] = 3.0
# model.F_eff['p'] = 5.0
#
#
# lb = {'a': 1, 'b': 1, 'c': 0, 'e': 0, 'p': 0, 'g': 0}
# ub = {'a': infinity, 'b': infinity, 'c': infinity, 'e': infinity, 'p': 4.763, 'g': infinity}
#
# def f_bounds(model, i):
#     return (lb[i], ub[i])
#
#
# model.F = Var(model.i, initialize=0, bounds=f_bounds)
# model.F['a'] = 10.0
# model.F['b'] = 20.0
# model.F['g'] = 1.0
# model.F['p'] = 0.5
#
# model.F_r = Var(model.i, initialize=0, bounds=(0, infinity))
# model.F_sum_eff = Var(initialize=52, bounds=(1, infinity))
# model.F_purge = Var(initialize=0.0, bounds=(0, infinity))
#
# model.n_eff = Var(initialize=0., bounds=(0, 1))
# model.V = Var(initialize=0.06, bounds=(0.03, 0.1))
# model.T = Var(initialize=5.80, bounds=(5.8, 6.8))
#
# # ********************** External function section***********************************************
# a2 = 2.5962e12
# a3 = 9.6283e15
# rho = 50
#
# def blackbox_2(comp_1, comp_2, temp, vol):
#     return a2 * comp_1 * comp_2 * vol * rho * exp(-150 / temp)
# bb_2 = ExternalFunction(blackbox_2)
#
# def blackbox_3(comp_1, comp_2, temp, vol):
#     return a3 * comp_1 * comp_2 * vol * rho * exp(-200 / temp)
# bb_3 = ExternalFunction(blackbox_3)
#
# model.r2 = Var(initialize=0, bounds=(0, infinity))
# model.rex_c2 = Constraint(expr= model.r2 == bb_2(model.x['b'], model.x['c'], model.T, model.V))
#
# model.r3 = Var(initialize=0, bounds=(0, infinity))
# model.rex_c3 = Constraint(expr= model.r3 == bb_3(model.x['p'], model.x['c'], model.T, model.V))
#
# model.r1 = Var(initialize=3)
# model.T2 = Var(initialize = 0.147, bounds=(0.147, 1.725))
# model.j = Set(initialize=[1])
#
# model.q = Param(model.j, mutable=True)
# model.q[1] = 120
#
# model.a = Param(model.j, mutable=True)
# model.a[1] = 5.9755e9
#
# model.ac = Var(model.j, initialize=0)
#
# model.kf = Var(model.j)
# model.kf[1] = 6.18
#
#
# def k_constraint(model, j):
#     return model.ac[j] == log(model.a[j]) - model.q[j] * model.T2
#
# model.re1 = Constraint(model.j, rule=k_constraint, doc='K-prime evaluation')
#
# model.re2 = Constraint(expr=model.T * model.T2 == 1)
#
# def kf_constraint(model, j):
#     return model.kf[j] == exp(model.ac[j])
#
# model.re3 = Constraint(model.j, rule=kf_constraint, doc='exponential term')
#
# def reaction_rate_1_expression(model):
#     return model.r1 == model.kf[1] * model.x['a'] * model.x['b'] * model.V * model.rho
#
# model.re4 = Constraint(rule=reaction_rate_1_expression, doc='r1 expression')
# # ***********************************************************************************************
#
# model.rbe_c1 = Constraint(expr=model.F_eff['a'] - model.F['a'] - model.F_r['a'] + model.r1 == 0)
#
# model.rbe_c2 = Constraint(expr=model.F_eff['b'] - model.F['b'] - model.F_r['b'] + model.r1 + model.r2 == 0)
#
# model.rbe_c3 = Constraint(expr=model.F_eff['c'] - model.F_r['c'] - 2 * model.r1 + model.r2 * 2 + model.r3 == 0)
#
# model.rbe_c4 = Constraint(expr=model.F_eff['e'] - model.F_r['e'] - model.r2 * 2  == 0)
#
# model.rbe_c5 = Constraint(expr=model.F_eff['p'] - 0.1 * model.F_r['e'] - model.r2 + model.r3 * 0.5 == 0)
#
# model.rbe_c6 = Constraint(expr=model.F_eff['g'] == model.r3 * 1.5)
#
# model.rbe_c7 = Constraint(expr=model.F_sum_eff - sum(model.F_eff[m] for m in model.i) == 0)
#
# def mole_frac_eval(model, i):
#     return model.F_eff[i] == model.F_sum_eff * model.x[i]
#
# model.rbe_c8 = Constraint(model.i, rule=mole_frac_eval, doc='Define mole fractions')
#
# model.rbe_c9 = Constraint(expr= -(model.F_sum_eff - 1) <= 0)
#
# model.ws_c1 = Constraint(expr=model.F['g'] - model.F_eff['g'] == 0)
#
# model.ps_c1 = Constraint(expr=model.F['p'] == model.F_eff['p'] - 0.1 * model.F_eff['e'])
#
# model.pg_c1 = Constraint(expr=model.F_purge == (model.F_eff['a'] + model.F_eff['b'] + model.F_eff['c'] + 1.1 * model.F_eff['e']) * model.n_eff)
#
# def recycle_eval(model, i):
#     return model.F_r[i] == model.F_eff[i] * (1 - model.n_eff)
#
# model.rs_c1 = Constraint(model.i, rule=recycle_eval, doc='Define recycle stream')
#
#
# model.ROI = Objective(
#     expr= -((2207 * model.F['p'] + 50 * model.F_purge - 168 * model.F['a'] - 252 * model.F['b'] - 2.22 * model.F_sum_eff - 84 * model.F['g'] - 60 * model.V * model.rho) / (6 * model.V *  model.rho)), sense=minimize
#                       )
#
# instance = model
# optTRF = SolverFactory('trustregion')
# result = optTRF.solve(instance, [bb_2, bb_3])
# instance.display()
