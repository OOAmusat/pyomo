from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyutilib.math import infinity


def hs100lnp_problem():
    # # 1. CUTEr problem hs100lnp
    m = ConcreteModel()
    m.j = Set(initialize=[1, 2, 3, 4, 5, 6, 7])

    init_x = {1: 1, 2: 2, 3:0, 4:4, 5: 0, 6: 1, 7: 1}  # init_x = {1: 1, 2: 2, 3:0, 4:4, 5: 0, 6: 1, 7: 1}

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

    optTRF = SolverFactory('trustregion')
    # ep_i verifies that theta (the infeasibility) is low enough
    # criticality_check determmines whether a criticality update is required and controls sample radius shrinking
    # max_iter is the maximum number of iterations
    optTRF.solve(m, [bb], reduced_model_type=3, trust_radius=20, sample_radius=0.2, max_it = 100, criticality_check = 0.01, ep_i = 1e-6,  eta2=0.75, compatibility_penalty=1e-6)

    m.display()

def hs100lnp_problem_v2():
    # Same problem as above but implemented exactly as in Eason thesis
    m = ConcreteModel()
    m.j = Set(initialize=[1, 2, 3, 4])
    m.z = Var([1, 2], domain=Reals, initialize=1.)

    init_x = {1: 1, 2: 2, 3:4, 4: 0}
    def x_init(m, i):
        return (init_x[i])
    m.x = Var(m.j, initialize=x_init)

    def blackbox(a, b, c, d):
       return 127 - 2 * (a ** 2) - 3 * (b ** 4) - 4 * (c ** 2) - 5 * d
    bb = ExternalFunction(blackbox)

    q = bb(m.x[1], m.x[2], m.x[3], m.x[4])

    m.objfunc = Objective(expr=
                         ((m.x[1] - 10)**2 + 5*(m.x[2] - 12)**2 + (bb(m.x[1], m.x[2], m.x[3], m.x[4]))**4 + 3 * (m.x[3] - 11)**2 + 10 * m.x[4]**6 + 7 * m.z[1]**2 + m.z[2]**4 - 4 * m.z[1] * m.z[2] - 10 * m.z[1] - 8 * m.z[2])
                         )

    m.second_constraint = Constraint(expr= -4 * m.x[1] * m.x[1] - m.x[2] * m.x[2] + 3 * m.x[1] * m.x[2] - 2 * (bb(m.x[1], m.x[2], m.x[3], m.x[4])) * (bb(m.x[1], m.x[2], m.x[3], m.x[4])) - 5 * m.z[1] + 11 * m.z[2] == 0)

    optTRF = SolverFactory('trustregion')
    optTRF.solve(m, [bb])
    m.display()

def hs46_problem():
    # 1. CUTEr problem hs046
    m = ConcreteModel()
    m.j = Set(initialize=[1, 2, 3, 4, 5])

    init_x = {1: 0.707, 2: 1.75, 3:0.5, 4:2, 5: 2}  # init_x = {1: 1, 2: 2, 3:0, 4:4, 5: 0, 6: 1, 7: 1}

    def x_init(m, i):
        return (init_x[i])

    m.x = Var(m.j, initialize = x_init)

    def blackbox(a, b):
       return sin(a - b) - 1
    bb = ExternalFunction(blackbox)

    m.first_constraint = Constraint(expr= (m.x[1]**2) * m.x[4] + bb(m.x[4], m.x[5])  ==0)

    m.second_constraint = Constraint(expr= m.x[2] + (m.x[3]**4) * (m.x[4]**2) - 2 == 0)

    m.objfunc = Objective(expr=
                          ((m.x[1] - m.x[2])**2 + (m.x[3] - 1)**2 + (m.x[4] - 1)**4 + (m.x[5] - 1)**6)
                         )

    m.pprint()
    optTRF = SolverFactory('trustregion')
    optTRF.solve(m, [bb], reduced_model_type=0, trust_radius=10, sample_radius=0.2, max_it = 10, criticality_check = 0.01, ep_i = 1e-6,  eta2=0.75, compatibility_penalty=1e-6)
    m.display()

def hs107_problem_v2():
    # Nine blackbox inputs (nx = y1-y6, x5-x7), four outputs (bracket terms in first two constraints)
    m = ConcreteModel()
    m.i = Set(initialize=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    m.j = Set(initialize=[1, 2, 3, 4, 5, 6])

    c = (48.4/50.176) * sin(.25)
    d = (48.4/50.176) * cos(.25)

    init_x = {1: 0.8, 2: 0.8, 3:0.2, 4:0.2, 5: 1.0454, 6: 1.0454, 7:1.0454, 8:0, 9:0}  # init_x = {1: 1, 2: 2, 3:0, 4:4, 5: 0, 6: 1, 7: 1}

    def x_init(m, i):
        return (init_x[i])

    lb = {1: 0, 2: 0, 3: -infinity, 4: -infinity, 5: 0.90909, 6: 0.90909, 7: 0.90909, 8: -infinity, 9: -infinity}
    ub = {1: infinity, 2: infinity, 3: infinity, 4: infinity, 5: 1.0909, 6: 1.0909, 7: 1.0909, 8: infinity, 9: infinity}

    def x_bounds(m, i):
        return (lb[i], ub[i])

    m.x = Var(m.i, initialize = x_init, bounds=x_bounds)
    m.y = Var(m.j, initialize=0.5)

    def blackbox_1(comp_1, comp_2, comp_3, comp_4):
        return comp_1 * comp_2 * (d * comp_3 + c * comp_4)
    bb_1 = ExternalFunction(blackbox_1)

    m.y1 = Constraint(expr=m.y[1] == sin(m.x[8]))
    m.y2 = Constraint(expr=m.y[2] == cos(m.x[8]))
    m.y3 = Constraint(expr=m.y[3] == sin(m.x[9]))
    m.y4 = Constraint(expr=m.y[4] == cos(m.x[9]))
    m.y5 = Constraint(expr=m.y[5] == sin(m.x[8] - m.x[9]))
    m.y6 = Constraint(expr=m.y[6] == cos(m.x[8] - m.x[9]))


    m.first_constraint = Constraint(expr=0.4 - m.x[1] + (2 * c * m.x[5]**2) - bb_1(m.x[5], m.x[6], m.y[1], m.y[2]) -  bb_1(m.x[5], m.x[7], m.y[3], m.y[4]) == 0)
    m.second_constraint = Constraint(expr=0.4 - m.x[2] + (2 * c * m.x[6]**2) + bb_1(m.x[5], m.x[6], m.y[1], -m.y[2]) + bb_1(m.x[6], m.x[7], m.y[5], -m.y[6]) == 0)
    m.third_constraint = Constraint(expr=0.8 + (2 * c * m.x[7]**2) + m.x[5] * m.x[7] * (d* m.y[3] - c * m.y[4]) - m.x[6] * m.x[7] * (d * m.y[5] + c * m.y[6]) == 0)
    m.fourth_constraint = Constraint(expr=0.2 - m.x[3] + (2 * d * m.x[5]**2) + m.x[5] * m.x[6] * (c* m.y[1] - d * m.y[2]) + m.x[5] * m.x[7] * (c * m.y[3] - d * m.y[4]) == 0)
    m.fifth_constraint = Constraint(expr=0.2 - m.x[4] + (2 * d * m.x[6]**2) - m.x[5] * m.x[6] * (c* m.y[1] + d * m.y[2]) - m.x[6] * m.x[7] * (c * m.y[5] + d * m.y[6]) == 0)
    m.sixth_constraint = Constraint(expr=-0.337 + (2 * d * m.x[7]**2) - m.x[5] * m.x[7] * (c* m.y[3] + d * m.y[4]) + m.x[6] * m.x[7] * (c * m.y[5] - d * m.y[6]) == 0)

    m.objfunc = Objective(expr=
                          (3000 * m.x[1] + 1000 * m.x[1]**3 + 2000 * m.x[2] + 666.667 * m.x[2]**3)
                         )

    optTRF = SolverFactory('trustregion')
    optTRF.solve(m, [bb_1], reduced_model_type=3, trust_radius=10, sample_radius=0.2, max_it = 50, criticality_check = 0.01, ep_i = 1e-6,  eta2=0.75, compatibility_penalty=1e-6)
    m.display()

def hs107_problem():
    # CUTEr problem hs107
    # Nine blackbox inputs (nx = y1-y6, x5-x7), two outputs (x1 and x2 in first two constraints)

    m = ConcreteModel()
    m.i = Set(initialize=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    m.j = Set(initialize=[1, 2, 3, 4, 5, 6])

    c = (48.4/50.176) * sin(.25)
    d = (48.4/50.176) * cos(.25)

    init_x = {1: 0.8, 2: 0.8, 3:0.2, 4:0.2, 5: 1.0454, 6: 1.0454, 7:1.0454, 8:0, 9:0}  # init_x = {1: 1, 2: 2, 3:0, 4:4, 5: 0, 6: 1, 7: 1}

    def x_init(m, i):
        return (init_x[i])

    lb = {1: 0, 2: 0, 3: -infinity, 4: -infinity, 5: 0.90909, 6: 0.90909, 7: 0.90909, 8: -infinity, 9: -infinity}
    ub = {1: infinity, 2: infinity, 3: infinity, 4: infinity, 5: 1.0909, 6: 1.0909, 7: 1.0909, 8: infinity, 9: infinity}

    def x_bounds(m, i):
        return (lb[i], ub[i])

    m.x = Var(m.i, initialize = x_init, bounds=x_bounds)
    m.y = Var(m.j, initialize=0.5)

    def blackbox_1(comp_1, comp_2, comp_3, comp_4, comp_5, comp_6, comp_7):
        return 0.4 + (2 * c * comp_1 *comp_1) - comp_1 * comp_2 * (d * comp_4 + c * comp_5) - comp_1 * comp_3 * (d * comp_6 + c * comp_7)
    bb_1 = ExternalFunction(blackbox_1)


    def blackbox_2(comp_1, comp_2, comp_3, comp_4, comp_5, comp_6, comp_7):
        return 0.4 + (2 * c * comp_2 *comp_2) + comp_1 * comp_2 * (d * comp_4 - c * comp_5) + comp_2 * comp_3 * (d * comp_6 - c * comp_7)
    bb_2 = ExternalFunction(blackbox_2)

    m.y1 = Constraint(expr=m.y[1] == sin(m.x[8]))
    m.y2 = Constraint(expr=m.y[2] == cos(m.x[8]))
    m.y3 = Constraint(expr=m.y[3] == sin(m.x[9]))
    m.y4 = Constraint(expr=m.y[4] == cos(m.x[9]))
    m.y5 = Constraint(expr=m.y[5] == sin(m.x[8] - m.x[9]))
    m.y6 = Constraint(expr=m.y[6] == cos(m.x[8] - m.x[9]))


    m.first_constraint = Constraint(expr= - m.x[1] + bb_1(m.x[5], m.x[6], m.x[7], m.y[1], m.y[2], m.y[3], m.y[4]) == 0)
    m.second_constraint = Constraint(expr= - m.x[2] +  bb_2(m.x[5], m.x[6], m.x[7], m.y[1], m.y[2], m.y[5], m.y[6])== 0)
    m.third_constraint = Constraint(expr=0.8 + (2 * c * m.x[7]**2) + m.x[5] * m.x[7] * (d* m.y[3] - c * m.y[4]) - m.x[6] * m.x[7] * (d * m.y[5] + c * m.y[6]) == 0)
    m.fourth_constraint = Constraint(expr=0.2 - m.x[3] + (2 * d * m.x[5]**2) + m.x[5] * m.x[6] * (c* m.y[1] - d * m.y[2]) + m.x[5] * m.x[7] * (c * m.y[3] - d * m.y[4]) == 0)
    m.fifth_constraint = Constraint(expr=0.2 - m.x[4] + (2 * d * m.x[6]**2) - m.x[5] * m.x[6] * (c* m.y[1] + d * m.y[2]) - m.x[6] * m.x[7] * (c * m.y[5] + d * m.y[6]) == 0)
    m.sixth_constraint = Constraint(expr=-0.337 + (2 * d * m.x[7]**2) - m.x[5] * m.x[7] * (c* m.y[3] + d * m.y[4]) + m.x[6] * m.x[7] * (c * m.y[5] - d * m.y[6]) == 0)

    m.objfunc = Objective(expr=
                          (3000 * m.x[1] + 1000 * m.x[1]**3 + 2000 * m.x[2] + 666.667 * m.x[2]**3)
                         )

    m.pprint()
    optTRF = SolverFactory('trustregion')
    optTRF.solve(m, [bb_1, bb_2], reduced_model_type=2, trust_radius=10, sample_radius=0.5, max_it = 50, criticality_check = 0.01, ep_i = 1e-6, eta2=0.75, compatibility_penalty=1e-6, filter_flag=True)
    m.display()

def hs77_problem():
    # # CuTER problem hs77
    m = ConcreteModel()
    m.z = Var(range(3), domain=Reals, initialize=2.)
    m.x = Var(range(2), initialize=2.)
    m.x[1] = 1.0

    def blackbox(a,b):
       return sin(a-b)
    bb = ExternalFunction(blackbox)

    m.obj = Objective(
       expr=(m.z[0]-1.0)**2 + (m.z[0]-m.z[1])**2 + (m.z[2]-1.0)**2 \
           + (m.x[0]-1.0)**4 + (m.x[1]-1.0)**6 # + m.bb(m.x[0],m.x[1])
    )
    m.c1 = Constraint(expr=m.x[0] * m.z[0]**2 + bb(m.x[0],m.x[1])== 2*sqrt(2.0))
    m.c2 = Constraint(expr=m.z[2]**4 * m.z[1]**2 + m.z[1] == 8+sqrt(2.0))

    # m.pprint()
    optTRF = SolverFactory('trustregion')
    optTRF.solve(m, [bb])
    m.display()

def hs75_problem():
    # # 4. CuTER problem hs74, 75
    a = 0.48 # 0.55
    m = ConcreteModel()
    m.i = Set(initialize=[1, 2, 3, 4])

    lb = {1: 0, 2: 0, 3: -a, 4: -a}
    ub = {1: 1200, 2: 1200, 3: a, 4: a}

    def x_bounds(m, i):
        return (lb[i], ub[i])

    m.x = Var(m.i, initialize=0., bounds=x_bounds)

    def blackbox(a, b):
       return sin(a + b - 0.25)
    bb = ExternalFunction(blackbox)

    m.con1 = Constraint(expr= -m.x[4] + m.x[3] - a <= 0)
    m.con2 = Constraint(expr= - m.x[3] + m.x[4] - a <= 0)
    m.con3 = Constraint(expr=1000*sin(-m.x[3] - 0.25) + 1000*sin(-m.x[4] - 0.25) + 894.8 - m.x[1] == 0)
    m.con4 = Constraint(expr=1000*sin(m.x[3] - 0.25) + 1000*bb(m.x[3], -m.x[4]) + 894.8 - m.x[2] == 0)
    m.con5 = Constraint(expr=1000*sin(m.x[4] - 0.25) + 1000*bb(-m.x[3], m.x[4]) + 1294.8 == 0)


    m.obj =  Objective(expr=3*m.x[1] + (1.0e-6 * m.x[1]**3) + 2 * m.x [2] + ((1/3) * 2.0e-6 * m.x[2] ** 3))

    m.pprint()
    optTRF = SolverFactory('trustregion')
    optTRF.solve(m, [bb], reduced_model_type=2, trust_radius=1000, sample_radius=0.2, max_it = 50, criticality_check = 0.01, ep_i = 1e-6, eta2=0.75, compatibility_penalty=1e-6)
    m.display()

def hs81_problem():
    # 4. CuTER problem hs81
    m = ConcreteModel()
    m.i = Set(initialize=[1, 2, 3, 4, 5])

    lb = {1: -2.3, 2: -2.3, 3: -3.2, 4: -3.2, 5: -3.2}
    ub = {1: 2.3, 2: 2.3, 3: 3.2, 4: 3.2, 5: 3.2}

    def x_bounds(m, i):
        return (lb[i], ub[i])

    init_x = {1: -2, 2: 2, 3: 2, 4: -1, 5: -1}

    def x_init(m, i):
        return (init_x[i])

    m.x = Var(m.i, initialize=x_init, bounds=x_bounds)

    def blackbox(a, b):
       return a**3 + b**3 + 1
    bb = ExternalFunction(blackbox)

    m.con1 = Constraint(expr=m.x[1]**2 + m.x[2]**2 + m.x[3]**2 + m.x[4]**2 + m.x[5]**2 -10 == 0)
    m.con2 = Constraint(expr= m.x[2] * m.x[3] - 5 * m.x[4]*m.x[5] == 0)
    m.con3 = Constraint(expr= bb(m.x[1], m.x[2]) == 0)


    m.obj =  Objective(expr= (exp(m.x[1] * m.x[2] * m.x[3] * m.x[4] * m.x[5]) - 0.5 * (m.x[1] **3 + m.x[2] ** 3 + 1)**2))

    m.pprint()
    optTRF = SolverFactory('trustregion')
    optTRF.solve(m, [bb], reduced_model_type=3, trust_radius=5, sample_radius=0.2, max_it = 250, criticality_check = 0.01, ep_i = 1e-6, eta2=0.75, compatibility_penalty=1e-6)
    m.display()


# hs46_problem()
# hs107_problem()
# hs107_problem_v2()
# hs77_problem()
# hs75_problem()
# hs81_problem()
# hs100lnp_problem()
hs100lnp_problem_v2()