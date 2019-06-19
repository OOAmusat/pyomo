from pyomo.contrib.polynomial_regression import *
from pyomo.contrib.sampling import sampling as sp
from pyomo.common.fileutils import PYOMO_ROOT_DIR
import pandas as pd
import numpy as np
import os

os.path.join(PYOMO_ROOT_DIR, 'contrib', 'surrogates', 'examples', 'data_files')

# Load XY data from high fidelity model from tab file using Pandas. Y data must be in the last column.
data = pd.read_csv('three_humpback_data_v4.csv', header=0, index_col=0)

# data = pd.read_excel('matyas_function.xls', header=0, index_col=0)
# data = pd.read_csv('six_hump_function_data.tab', sep='\t', header=0, index_col=0)
# data = pd.read_csv('cozad_function_data_v2.txt', sep='\s+', header=0, index_col=0)
# data = pd.read_csv('mass_spring_data.txt', sep='\s+', header=None, index_col=None)

b = sp.LatinHypercubeSampling(data, 75)
c = b.lh_sample_points()

# Carry out polynomial regression, feeding in the original data and the sampled data
d = PolynomialRegression(data, c, maximum_polynomial_order=10, max_iter=20, multinomials=1, solution_method='pyomo')
results_object = d.polynomial_regression_fitting([ np.sin(c[:, 0]), np.cos(c[:, 0]), np.sin(c[:, 1]), np.cos(c[:, 1]) ])
