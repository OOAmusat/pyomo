import sys
import io
import numpy as np
from idaes.surrogate.pysmo import sampling, polynomial_regression, kriging


class ROMType:
    interpolation = 0
    linear0 = 1
    linear1 = 2
    quadratic = 3
    kriging = 4


def evaluateRx(self, surr_obj, x):
    """
    ``evaluateRx`` evaluates the values of the surrogates at a point x

    :param surr_obj: Python objects containing surrogate expressions
    :param x: Current point at which surrogate(s) will be evaluated
    :return: rx, the surrogate predictions at point x
    """
    rx = np.zeros((self.ly,))
    for i in range(0, self.ly):
        values = []
        for j in self.exfn_xvars_ind[i]:
            values.append(x[j])

        rx[i,] = surr_obj[i][0](np.array([values]))
    return rx


def surrogate_obj_constraints(self, ly, surr_eqs):
    """
    ``surrogate_obj_constraints`` generates the pyomo expressions and objective function associated with the surrogate.

    The function returns the objective function of the compatibility check ||y-r(w)||  and the surrogate constraints for the TPSPk problem.
    See Eason et al. (2016) for details.

    :param ly: Number of external functions for which surrogates exist
    :param surr_eqs: List of Pyomo surrogate expressions
    :return: surr_obj, the objective of compatibility check problem (||y-r(w)||
             surr_con, a list containing the surrogate-dependent constraints y-r(w) for the TPSPK problem
    """
    surr_obj = 0
    surr_con = []
    model = self.TRF
    for i in range(0, self.ly):
        surr_obj += (surr_eqs[i][0] - model.y[i + 1]) ** 2
        surr_con.append(model.y[i + 1] - surr_eqs[i][0])
    return surr_obj, surr_con


def buildROM(self, x, radius_base):
    """
    ``buildROM`` generates the surrogate models r(w) for the external functions

    :param x: Current point around which the surrogate must be generated
    :param radius_base: Sample radius. The sample points for surrogate generation will be generated between (x+radius_base) and (x-radius_base)
    :return: a number of surrogate related information, including:
             surrogate_objects: objects containing the surrogate expressions
             surrogate_objective: Objective function of the compatibility problem, ||y-r(w)|
             surrogate_constraints: Surrogate related constraints for the TPSPk and criticality problems, y-r(w)
             y1: True output values of the blackbox at x
    """

    y1 = self.evaluateDx(x)
    rom_params = []

    if (self.romtype == ROMType.linear0 or self.romtype == ROMType.linear1 or self.romtype == ROMType.quadratic or self.romtype == ROMType.kriging):
        # Trap all print to screens from sampling and PR scripts
        text_trap = io.StringIO()
        sys.stdout = text_trap

        # Create samples
        radius = radius_base  # * scale[j]
        x_lo = x - radius
        x_up = x + radius

        list_of_surrogates = []  # Will contain the surrogate parameters
        y_surrogates = []  # Will contain the output predictions from the surrogates when required
        surrogate_expressions = []  # Will contain the list of surrogate expressions
        surrogate_objects = []

        # For all external functions (verify!):
        for k in range(0, self.ly):
            surrogate_expressions.append([])
            surrogate_objects.append([])
            x_rel = []
            x_lo_rel = []
            x_up_rel = []
            for j in range(0, len(self.exfn_xvars_ind[k])):
                x_rel.append(x[self.exfn_xvars_ind[k][j]])
                x_lo_rel.append(x_lo[self.exfn_xvars_ind[k][j]])
                x_up_rel.append(x_up[self.exfn_xvars_ind[k][j]])
            x_bounds = [x_lo_rel, x_up_rel]

            #############################################################
            # Fix fraction for training and calculate number of samples based on number of features
            tr_split = 0.8
            if self.romtype == ROMType.linear0:
                num_sp = int(np.around(((len(x_lo_rel) + 1) * (1 / tr_split)))) 
            elif self.romtype == ROMType.linear1:
                num_sp = int(np.around(((0.5 * (len(x_lo_rel) + len(x_lo_rel) ** 2) + 1) * (1 / tr_split))))
            elif self.romtype == ROMType.quadratic:
                num_sp = int(np.around(((0.5 * (3 * len(x_lo_rel) + len(x_lo_rel) ** 2) + 1) * (1 / tr_split))))
            elif self.romtype == ROMType.kriging:
                num_sp = 25  # len(x_lo_rel) + 3 # number of features + s.d + mean + reg_param

            # # Calculate number of samples as twice the number of features
            # tr_split = 0.8
            # if self.romtype == ROMType.linear:
            #     num_sp = int((len(x_lo_rel) + 1) * 2) - 2
            # elif self.romtype == ROMType.quadratic:
            #     num_sp = int(np.around(((0.5 * (3 * len(x_lo_rel) + len(x_lo_rel) ** 2) + 1) * (2)))) - 2
            # elif self.romtype == ROMType.kriging:
            #     num_sp = 25  # len(x_lo_rel) + 3 # number of features + s.d + mean + reg_param
            #############################################################

            region_sampling = sampling.HammersleySampling(x_bounds, number_of_samples=num_sp,
                                                          sampling_type="creation")  # random number of samples
            values = region_sampling.sample_points()
            x_rel = np.array(x_rel)
            values = np.concatenate((x_rel.reshape(1, x_rel.shape[0]), values), axis=0)
            x_up_rel = np.array(x_up_rel)
            values = np.concatenate((x_up_rel.reshape(1, x_up_rel.shape[0]), values), axis=0)

            # b. generate output from actual function
            fcn = self.TRF.external_fcns[k]._fcn
            y_samples = []
            for j in range(0, values.shape[0]):
                y_samples.append(fcn._fcn(*values[j, :]))
            y_samples = np.array(y_samples)
            if y_samples.ndim == 1:
                y_samples = y_samples.reshape(len(y_samples), 1)

            # c. Generate a surrogate for each output and store in list_of_surrogates
            number_bb_outputs = y_samples.shape[1]
            for i in range(0, number_bb_outputs):
                surrogate_predictions = []
                training_samples = np.concatenate((values, y_samples[:, i].reshape(y_samples.shape[0], 1)), axis=1)

                # Generate pyomo equations: collect index of terms in indx, collect terms from xvars, then generate expression
                indx = self.exfn_xvars_ind[k]
                surr_vars = []
                for p in range(0, len(indx)):
                    surr_vars.append(self.TRF.xvars[indx[p]])

                if self.romtype == ROMType.linear0:
                    call_surrogate_method = polynomial_regression.PolynomialRegression(training_samples,
                                                                                       training_samples,
                                                                                       maximum_polynomial_order=1,
                                                                                       multinomials=0,
                                                                                       number_of_crossvalidations=3,
                                                                                       solution_method="mle",
                                                                                       training_split=tr_split)
                    p = call_surrogate_method.get_feature_vector()
                    call_surrogate_method.set_additional_terms([])
                    results = call_surrogate_method.poly_training()
                    surrogate_expressions[k].append(results.generate_expression(surr_vars))
                    surrogate_objects[k].append(lambda u, call_surrogate_method=call_surrogate_method,
                                                       results=results: call_surrogate_method.poly_predict_output(
                        results, u))
                    list_of_surrogates.append(results.optimal_weights_array.flatten().tolist())
                if self.romtype == ROMType.linear1:
                    call_surrogate_method = polynomial_regression.PolynomialRegression(training_samples,
                                                                                       training_samples,
                                                                                       maximum_polynomial_order=1,
                                                                                       multinomials=1,
                                                                                       number_of_crossvalidations=3,
                                                                                       solution_method="mle",
                                                                                       training_split=tr_split)
                    p = call_surrogate_method.get_feature_vector()
                    call_surrogate_method.set_additional_terms([])
                    results = call_surrogate_method.poly_training()
                    surrogate_expressions[k].append(results.generate_expression(surr_vars))
                    surrogate_objects[k].append(lambda u, call_surrogate_method=call_surrogate_method,
                                                       results=results: call_surrogate_method.poly_predict_output(
                        results, u))
                    list_of_surrogates.append(results.optimal_weights_array.flatten().tolist())
                elif self.romtype == ROMType.quadratic:
                    call_surrogate_method = polynomial_regression.PolynomialRegression(training_samples,
                                                                                       training_samples,
                                                                                       maximum_polynomial_order=2,
                                                                                       multinomials=1,
                                                                                       number_of_crossvalidations=3,
                                                                                       solution_method="mle",
                                                                                       training_split=tr_split)
                    p = call_surrogate_method.get_feature_vector()
                    call_surrogate_method.set_additional_terms([])
                    results = call_surrogate_method.poly_training()
                    surrogate_expressions[k].append(results.generate_expression(surr_vars))
                    surrogate_objects[k].append(lambda u, call_surrogate_method=call_surrogate_method,
                                                       results=results: call_surrogate_method.poly_predict_output(
                        results, u))
                    # surrogate_objects[k].append([call_surrogate_method, results])
                    if results.polynomial_order == 1:
                        no_comb_terms = int(
                            0.5 * len(self.exfn_xvars_ind[k]) * (len(self.exfn_xvars_ind[k]) - 1))
                        adjusted_vec_coeffs = np.zeros((1 + 2 * len(self.exfn_xvars_ind[k]) + no_comb_terms, 1))
                        adjusted_vec_coeffs[0, 0] = results.optimal_weights_array[0, 0]
                        adjusted_vec_coeffs[1:len(self.exfn_xvars_ind[k]) + 1,
                        0] = results.optimal_weights_array[1:len(self.exfn_xvars_ind[k]) + 1, 0]
                        adjusted_vec_coeffs[-no_comb_terms:, 0] = results.optimal_weights_array[-no_comb_terms:, 0]
                        list_of_surrogates.append(adjusted_vec_coeffs.flatten().tolist())
                    else:
                        list_of_surrogates.append(results.optimal_weights_array.flatten().tolist())
                elif self.romtype == ROMType.kriging:
                    call_surrogate_method = kriging.KrigingModel(training_samples, regularization=True,
                                                                 numerical_gradients=False)
                    p = call_surrogate_method.get_feature_vector()
                    results = call_surrogate_method.kriging_training()
                    surrogate_expressions[k].append(results.kriging_generate_expression(surr_vars))
                    surrogate_objects[k].append(lambda u, call_surrogate_method=call_surrogate_method,
                                                       results=results: call_surrogate_method.kriging_predict_output(
                        results, u))
                    list_of_surrogates.append(results.optimal_weights.flatten().tolist())

            # y_surrogates.append(surrogate_predictions)

            # Return in form of the original ROM function
            rom_params = list_of_surrogates
        # End text trap
        sys.stdout = sys.__stdout__

    elif (self.romtype == ROMType.interpolation):

        def interpolation_expression(coeffs, vars, vals):
            expr = coeffs[0]
            for i in range(1, len(coeffs)):
                expr += coeffs[i] * (vars[i - 1] - vals[i - 1])
            return expr

        def interpolation_evaluation(eqn, vars, x_data):
            from pyomo.environ import Objective, ConcreteModel, value
            md = ConcreteModel()
            md.o2 = Objective(expr=eqn)
            y_eq = np.zeros((x_data.shape[0], 1))
            for j in range(0, x_data.shape[0]):
                for i in range(0, len(vars)):
                    vars[i] = x_data[j, i]
            y_eq[j, 0] = value(md.o2([vars]))
            return y_eq

        list_of_surrogates = []  # Will contain the surrogate parameters
        y_surrogates = []  # Will contain the output predictions from the surrogates when required
        surrogate_expressions = []  # Will contain the list of surrogate expressions
        surrogate_objects = []

        # For all external functions (verify!):
        for k in range(0, self.ly):
            surrogate_expressions.append([])
            surrogate_objects.append([])
            rom_params.append([])
            rom_params[k].append(y1[k])

            # Generate pyomo equations: collect index of terms in indx, collect terms from xvars, then generate expression
            indx = self.exfn_xvars_ind[k]
            surr_vars = []
            for p in range(0, len(indx)):
                surr_vars.append(self.TRF.xvars[indx[p]])

            # Check if it works with Ampl
            fcn = self.TRF.external_fcns[k]._fcn
            values = [];
            for j in self.exfn_xvars_ind[k]:
                values.append(x[j])

            # Evaluate coefficients:  same as original implementation
            for j in range(0, len(values)):
                radius = radius_base  # * scale[j]
                values[j] = values[j] + radius
                y2 = fcn._fcn(*values)
                rom_params[k].append((y2 - y1[k]) / radius)
                values[j] = values[j] - radius

            # Generate expression and surrogate object
            surrogate_expressions[k].append(interpolation_expression(rom_params[k], surr_vars, values))
            surrogate_objects[k].append(
                lambda u, interpolation_evaluation=interpolation_evaluation, surr_vars=surr_vars, values=values:
                interpolation_evaluation(interpolation_expression(rom_params[k], surr_vars, values), surr_vars, u))
            list_of_surrogates.append(rom_params[k])

    surrogate_objective, surrogate_constraints = surrogate_obj_constraints(self, self.ly, surrogate_expressions)
    return rom_params, surrogate_objects, surrogate_objective, surrogate_constraints, y1

