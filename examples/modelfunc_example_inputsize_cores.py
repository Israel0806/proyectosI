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

def calculateMu(p,m1,m2):
    return min(m1+m2/p, 1)

def calculateRho(k, phi):
    return 1 + k*phi

# values [f, m1, m2, k]
def _func_speedup(param, freq, cores):
    """
    Model function to calculate the speedup without overhead.

    :param fparam: Actual parameters values
    :param p: Numbers of cores used on model data
    :param n: Problems size used on model data
    :return: calculated speedup value
    """

    phi = (freq / 1000000) / 2.6

    #(1 − mu) + rho mu
    up_arg=(1-calculateMu(1,param[1],param[2]))+calculateRho(param[3],phi)*calculateMu(1,param[1],param[2])

    # ((1 − mu_p ) + rho mu_p ) * (1 − f ) + f/p)
    leftSide = ((1- calculateMu(cores, param[1], param[2])) + calculateRho(param[3], phi)) * ((1-param[0])+ param[0]/cores)

    # rho mu_p
    rightSide = phi*calculateMu(cores, param[1], param[2])

    #Calculate max btw below args
    below_arg = max(leftSide, rightSide)

    speedup = up_arg / below_arg
    return speedup


def model(par, x):
    """
    Mathematical Model function to predict the measures values.

    :param par: Actual parameters values
    :param x: inputs array
    :param overhead: If should be considered the overhead
    :return: Dict with input array ('x') and predicted output array ('y')
    """
    pred = []
    for f, p in x:
        y_model = _func_speedup(par, f, p)
        pred.append(y_model)
    return {'x': x, 'y': pred}

def constraint_function(par, x_meas, **kwargs):
    """
    Constraint function that would be considered on model.

    :param par: Actual parameters values
    :param args: Positional arguments passed for objective
                 and constraint functions
    :return: If parameters are acceptable based on return functions
    """

    pred = model(par, x_meas)
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
    pred = model(par, x_meas)
    # print("Y_meas")
    # print(y_meas)
    # print(pred['y'])
    # print()
    return mean_squared_error(y_meas, pred['y'])