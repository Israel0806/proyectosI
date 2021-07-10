# -*- coding: utf-8 -*-
"""
    Model function example to use with parsecpy runmodel script.
    Model for variation of input size (n) and number of cores (p).

    Speedup:
        S = 1 / ( ( 1-f(p,n) ) + f(p,n)/p + Q(p,n) )

    Parallel Fraction:
        f(p,n) = max( min((f1) + (f2)/p + (f3)*(f4)^n,1 ),0 )

    Overhead:
        Q(p,n) = (f5) + ( (f6)*p )/( (f7)^n )

"""

import numpy as np
from numpy.lib import math
from sklearn.metrics import mean_squared_error

# values [B]
def _func_speedup(param, freq, cores, f_min, min_exec_time):
    """
    Model function to calculate the speedup without overhead.

    :param fparam: Actual parameters values
    :param p: Numbers of cores used on model data
    :param n: Problems size used on model data
    :param B1: T(1, f_min) * B
    :param B2: T(1, f_min) * (1 - B)
    :return: calculated speedup value
    :return: B1 * f_min / f + B2 * f_min / (f * t)
    """
    B1 = min_exec_time * param[0]
    B2 = min_exec_time * (1 - param[0])
    return B1 * f_min / freq + B2 * f_min / (freq * cores)


def model(par, x_meas, f_min, min_exec_time):
    """
    Mathematical Model function to predict the measures values.

    :param par: Actual parameters values
    :param x: inputs array
    :param overhead: If should be considered the overhead
    :return: Dict with input array ('x') and predicted output array ('y')
    """
    pred = []
    for f, p in x_meas:
        y_model = _func_speedup(par, f, p, f_min, min_exec_time)
        pred.append(y_model)
    return {'x': x_meas, 'y': pred}

def constraint_function(par, x_meas, **kwargs):
    """
    Constraint function that would be considered on model.

    :param par: Actual parameters values
    :param args: Positional arguments passed for objective
                 and constraint functions
    :return: If parameters are acceptable based on return functions
    """

    pred = model(par, x_meas, kwargs['f_min'], kwargs['min_exec_time'])
    # print(pred)
    y = pred['y']
    # x_meas es el training_size
    # pred es la prediccion del algoritmo
    is_feasable = np.min(y) > 1
    return is_feasable


# Par: Particulas [f, m1, m2 k] 0-1 0-10
# x_meas: frecuencia y cores a tomar en cuenta (ts)
# y_meas: 
def objective_function(par, x_meas, y_meas, **kwargs):
    """
    Objective function (target function) to minimize.

    :param par: Actual parameters values
    :param args: Positional arguments passed for objective
                 and constraint functions
    :return: Mean squared error between measures and predicts
    """
    pred = model(par, x_meas, kwargs['f_min'], kwargs['min_exec_time'])
    return mean_squared_error(y_meas, pred['y'])