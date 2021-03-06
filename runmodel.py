#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Module to run a model of a parsec application.

    Its possible define the number of threads to execute a model
    on a fast way; The modelfunc to represent the application should be
    provided by user on a python module file. Its possible, also, provide a
    overhead function to integrate the model

    usage: parsecpy_runmodel [-h] --config CONFIG -f PARSECPYFILEPATH
                             [-p PARTICLES] [-x MAXITERATIONS]
                             [-l LOWERVALUES] [-u UPPERVALUES]
                             [-n PROBLEMSIZES] [-o OVERHEAD] [-t THREADS]
                             [-r REPETITIONS] [-c CROSSVALIDATION]
                             [-v VERBOSITY]

    Script to run modelling algorithm to predict a parsec application output

    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Filepath from Configuration file configurations
                            parameters
      -p PARSECPYDATAFILEPATH, --parsecpydatafilepath PARSECPYDATAFILEPATH
                            Path from input data file from Parsec specificated
                            package.
      -f FREQUENCIES, --frequency FREQUENCIES
                            List of frequencies (KHz). Ex: 2000000, 2100000
      -n PROBLEMSIZES, --problemsizes PROBLEMSIZES
                            List of problem sizes to model used. Ex:
                            native_01,native_05,native_08
      -o OVERHEAD, --overhead OVERHEAD
                            If it consider the overhead
      -t THREADS, --threads THREADS
                            Number of Threads
      -r REPETITIONS, --repetitions REPETITIONS
                            Run with repetitions to find the best model
      -c CROSSVALIDATION, --crossvalidation CROSSVALIDATION
                            If run the cross validation of modelling
      -m MEASURESFRACTION, --measuresfraction MEASURESFRACTION
                            Fraction of measures data to calculate the model
      -v VERBOSITY, --verbosity VERBOSITY
                            verbosity level. 0 = No verbose
    Example
        parsecpy_runmodel --config my_config.json
                          -p /var/myparsecsim.dat -c True -v 3
"""

import os
import sys
import json
import time
import argparse
from copy import deepcopy
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ShuffleSplit, KFold
# from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

from dataprocess import ParsecData
from pso import Swarm
from csa import CoupledAnnealer
from model import ParsecModel
from _common import argsparselist, argsparseintlist, argsparsefraction
from _common import data_detach, data_attach, measures_split_train_test

def argsparsevalidation():
    """
    Validation of script arguments passed via console.

    :return: argparse object with validated arguments.
    """

    parser = argparse.ArgumentParser(description='Script to run optimizer '
                                                 'modelling algorithm to '
                                                 'predict a parsec application '
                                                 'output')
    parser.add_argument('--config', required=True,
                        help='Filepath from Configuration file '
                             'configurations parameters')
    parser.add_argument('-p', '--parsecpydatafilepath',
                        help='Path from input data file from Parsec '
                             'specificated package.')
    parser.add_argument('-f', '--frequency', type=argsparseintlist,
                        help='List of frequencies (KHz). Ex: 2000000, 2100000')
    parser.add_argument('-n', '--problemsizes', type=argsparselist,
                        help='List of problem sizes to model used. '
                             'Ex: native_01,native_05,native_08')
    parser.add_argument('-s', '--serialfrequencyfixed', type=bool,
                        help='If it considers the serial time at fixed frequency')
    parser.add_argument('-t', '--threads', type=int,
                        help='Number of Threads')
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('-r', '--repetitions', type=int,
                        help='Number of Repetitions')
    group.add_argument('-c', '--crossvalidation', type=bool,
                       help='If run the cross validation of modelling')
    group.add_argument('-m', '--measuresfraction', type=int,
                       help='Number of points to use on model train')
    parser.add_argument('-v', '--verbosity', type=int,
                        help='verbosity level. 0 = No verbose')
    args = parser.parse_args()
    return args


def main(config, modelFun=False):
    """
    Main function executed from console run.

    """
    if 'repetitions' not in config.keys():
        config['repetitions'] = 1
    best_repetition = 1

    if 'serialfrequencyfixed' not in config.keys():
        config['serialfrequencyfixed'] = False

    if config['algorithm'] in ['pso', 'csa']:
        if not os.path.isfile(config['modelcodefilepath']):
            print('Error: You should inform the correct module of '
                  'objective function to model')
            sys.exit()

    if not os.path.isfile(config['parsecpydatafilepath']):
        print('Error: You should inform the correct parsecpy measures file')
        sys.exit()

    parsec_exec = ParsecData(modelFun, config['parsecpydatafilepath'])
    # print(parsec_exec)
    times, xspeedup_mod, measure = parsec_exec.speedups(serialfrequencyfixed=config['serialfrequencyfixed'])     #sequential_time/parallel_time

    # Get f_min and min_exec_time
    f_min = None
    max_exec_time = None
    if modelFun:  #DeSensi running
        m_data = data_detach(measure)
    else:
        m_data = data_detach(xspeedup_mod)

    min_index_freq = np.argmin(m_data['x'][:,0])
    f_min = m_data['x'][:,0][min_index_freq]

    """
    f_min: Minimum frequency
    min_index_freq: index indicanting config with minimum frequency
    max_exec_time: Maximum execution time
    min_exec_time: Minimum execution time
    exec_time_f_min_core_min: Execution time of minimum frequency and 1 core
    """
    for index, _config in enumerate(m_data['x']):
        # T(1, f_min)
        if str(_config[0]) == str(f_min) and str(_config[1]) == '1':
            exec_time_f_min_core_min = m_data['y'][index]

    min_index_time = np.argmin(m_data['y'])
    max_index_time = np.argmax(m_data['y'])

    max_exec_time = m_data['y'][max_index_time]
    min_exec_time = m_data['y'][min_index_time]
    kwargsmodel = {}
    kwargsmodel['f_min'] = f_min
    kwargsmodel['min_exec_time'] = exec_time_f_min_core_min

    input_sizes = []
    if 'size' in measure.dims:
        input_sizes = measure.attrs['input_sizes']
        input_ord = []
        if 'problemsizes' in config.keys():
                for i in config['problemsizes']:
                    if i not in input_sizes:
                        print('Error: Measures not has especified sizes')
                        sys.exit()
                    input_ord.append(input_sizes.index(i)+1)
                measure = measure.sel(size=sorted(input_ord))
                measure.attrs['input_sizes'] = sorted(config['problemsizes'])

    if 'frequency' in measure.dims:
        frequencies = measure.coords['frequency']
        if 'frequency' in config.keys():
                for i in config['frequency']:
                    if i not in frequencies:
                        print('Error: Measures not has especified frequencies')
                        sys.exit()
                measure = measure.sel(size=sorted(config['frequencies']))

    measure_detach = data_detach(measure)
    if 'measuresfraction' in config.keys():
        # xy_train_test = train_test_split(measure_detach['x'],
        #                                  measure_detach['y'],
        #                                  train_size=config['measuresfraction'])
        if 'powerOfTwo' in config.keys():
            xy_train_test = measures_split_train_test(measure,
                                                    train_size=config[
                                                        'measuresfraction'], onlyPowerOfTwo=True)
        else:    
            xy_train_test = measures_split_train_test(measure,
                                                    train_size=config[
                                                        'measuresfraction'])
        # TS configurations
        # configuraciones
        x_sample_train = xy_train_test[0]
        # los speedups reales
        y_sample_train = xy_train_test[2]
        # All speedups
        # todas las configuraciones
        x_sample_test = xy_train_test[1]
        # todos los speedups
        y_sample_test = xy_train_test[3]
    else:
        x_sample_train = measure_detach['x']
        y_sample_train = measure_detach['y']
        x_sample_test = measure_detach['x']
        y_sample_test = measure_detach['y']

    starttime = time.time()
    print('\nAlgorithm Execution...')

    if config['algorithm'] in ['svr', 'tree', 'neural']:
        measure_ml = measure.copy()
        measure_ml.coords['frequency'] = measure_ml.coords['frequency']/1e6
        measure_ml_detach = data_detach(measure_ml)
        scaler = StandardScaler()
        scaler.fit(measure_ml_detach['x'])
        feature_standardized = scaler.transform(measure_ml_detach['x'])
        measure_ml_standardized = data_attach({'x': feature_standardized,
                                               'y': measure_ml_detach['y']},
                                              measure_ml_detach['dims'])
        for j in range(config['repetitions']):
            print('Calculating model: Repetition=%d' % (j+1))
            # Checks if there is a training size sample for PSO
            if 'measuresfraction' in config.keys():
                # xy_train_test = train_test_split(measure_ml_detach['x'],
                #                                  measure_ml_detach['y'],
                #                                  train_size=config[
                #                                      'measuresfraction'])
                
                xy_train_test = measures_split_train_test(measure_ml,
                                                          train_size=config[
                                                              'measuresfraction'])
                x_sample_train = xy_train_test[0]
                y_sample_train = xy_train_test[2]
                x_sample_test = xy_train_test[1]
                y_sample_test = xy_train_test[3]
            else:
                x_sample_train = measure_ml_detach['x']
                y_sample_train = measure_ml_detach['y']
                x_sample_test = measure_ml_detach['x']
                y_sample_test = measure_ml_detach['y']
            if config['algorithm'] == 'svr':
                # print("X_SAMPLE")
                # print(x_sample_test)
                # print("Y_SAMPLE")
                # print(y_sample_train)
                gs_ml = GridSearchCV(SVR(),
                                      cv=config['crossvalidation-folds'],
                                      param_grid={"C": config['c_grid'],
                                                  "gamma": config['gamma_grid']})
                gs_ml.fit(x_sample_train, y_sample_train)
                best_params = gs_ml.best_params_
                print("best_params")
                print(best_params)
                for i, v in best_params.items():
                    config[i] = v
            elif config['algorithm'] == 'krr':                
                gs_ml = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                                     param_grid={"alpha": alpha,
                                                 "gamma": gamma})
                gs_ml.fit(x_sample_train, y_sample_train)
            elif config['algorithm'] == 'tree':
                gs_ml = DecisionTreeRegressor(random_state=0, max_depth=4)
                gs_ml.fit(x_sample_train, y_sample_train)
            elif config['algorithm'] == 'neural':
                alpha = 10.0 ** -np.arange(1, 7)
                gs_ml = make_pipeline(StandardScaler(),
                                      GridSearchCV(MLPRegressor(solver='lbfgs',
                                                                max_iter=2000,
                                                                random_state=0),
                                                   cv=3, param_grid={"alpha": alpha}))
                gs_ml.fit(x_sample_train, y_sample_train)
            
            y_predict = gs_ml.predict(x_sample_test)
            error = mean_squared_error(y_sample_test, y_predict)
            #errorrel = 100*error/np.mean(y_sample_test)
            # MSPE - Mean Square Percentage Error
            errorrel = 100 * np.sum(((y_predict - y_sample_test) / y_sample_test) ** 2) / np.size(y_sample_test)
            print('\n\n***** %s - Modelling Results! *****\n' % config['algorithm'].upper())
            print('Error: %.8f \nPercentual Error (Measured Mean): %.2f %%' %
                  (error, errorrel))
            y_model = data_attach({'x': measure_ml_detach['x'],
                                   'y': gs_ml.predict(measure_ml_detach['x'])},
                                  measure_ml_detach['dims'])

            # kf = KFold(n_splits=10, shuffle=True)
            # scores = cross_validate(gs_ml, x_sample_train, y_sample_train,
            #                         scoring='neg_mean_squared_error',
            #                         cv=kf, return_train_score=False)
            # if config['verbosity'] > 1:
            #     print(" ** Cross Validate Scores: ")
            #     print(scores)

            y_model.coords['frequency'] = y_model.coords[
                                                 'frequency'] * 1e6
            measure_ml.coords['frequency'] = measure_ml.coords[
                                                 'frequency'] * 1e6

            model = ParsecModel(measure=measure_ml, y_model=y_model,
                                berr=error, berr_rel=errorrel,
                                modelexecparams=config,
                                modelresultsfolder=config['resultsfolder'])
            if j == 0:
                model_best = deepcopy(model)
            else:
                if model.error < model_best.error:
                    model_best = deepcopy(model)
                    best_repetition = j+1
    else:
        for j in range(config['repetitions']):
            print('Calculating model: Repetition=%d' % (j+1))
            if config['algorithm'] == 'pso':
                optm = Swarm(config['lowervalues'], config['uppervalues'],
                             parsecpydatafilepath=config['parsecpydatafilepath'],
                             modelcodefilepath=config['modelcodefilepath'],
                             size=config['size'], w=config['w'],
                             c1=config['c1'], c2=config['c2'],
                             maxiter=config['maxiter'],
                             threads=config['threads'],
                             verbosity=config['verbosity'],
                             x_meas=x_sample_train, y_meas=y_sample_train,
                             kwargs=kwargsmodel)
            elif config['algorithm'] == 'csa':
                initial_state = np.array([np.random.uniform(size=config['dimension'])
                                          for _ in range(config['size'])])
                optm = CoupledAnnealer(initial_state,
                                       parsecpydatafilepath=config['parsecpydatafilepath'],
                                       modelcodefilepath=config['modelcodefilepath'],
                                       size=config['size'],
                                       steps=config['steps'],
                                       update_interval=config['update_interval'],
                                       tgen_initial=config['tgen_initial'],
                                       tgen_upd_factor=config['tgen_upd_factor'],
                                       tacc_initial=config['tacc_initial'],
                                       alpha=config['alpha'],
                                       desired_variance=config['desired_variance'],
                                       lowervalues=config['lowervalues'],
                                       uppervalues=config['uppervalues'],
                                       threads=config['threads'],
                                       verbosity=config['verbosity'],
                                       x_meas=x_sample_train,
                                       y_meas=y_sample_train,
                                       kwargs=kwargsmodel)
            else:
                print('Error: You should inform the correct algorithm to use')
                sys.exit()
            error, solution = optm.run()
            model = ParsecModel(bsol=solution,
                                berr=error,
                                measure=measure,
                                modelcodesource=optm.modelcodesource,
                                modelFun = modelFun,
                                kwargs = kwargsmodel,
                                modelexecparams=optm.get_parameters(),
                                modelresultsfolder=config['resultsfolder'])
            pred = model.predict(x_sample_test)
            # pred['x'] === ts
            model.error = mean_squared_error(y_sample_test, pred['y'])
            #model.errorrel = 100 * (model.error / np.mean(y_sample_test))
            # MSPE - Mean Square Percentage Error
            model.errorrel = 100*np.sum(((pred['y']-y_sample_test)/y_sample_test)**2)/np.size(y_sample_test)

            if j == 0:
                model_best = deepcopy(model)
            else:
                if model.error < model_best.error:
                    model_best = deepcopy(model)
                    best_repetition = j+1

        endtime = time.time()
        print('Best Model found on iteration = %d' % (best_repetition))
        print('Execution time = %.2f seconds' % (endtime - starttime))
        starttime = endtime

        print('\n\n***** %s - Modelling Results! *****\n' % config['algorithm'].upper())
        print('Error: %.8f \nPercentual Error (Measured Mean): %.2f %%' %
              (model_best.error,
               model_best.errorrel))
        if config['verbosity'] > 0:
            print('Best Parameters: \n', model_best.sol)
        if config['verbosity'] > 1:
            print('\nMeasured Speedup: \n', measure)
            print('\nModeled Speedup: \n', model_best.y_model)

        print('\n***** Modelling Done! *****\n')

        if config['crossvalidation']:
            print('\n\n***** Starting cross validation! *****\n')
            starttime = time.time()
            validation_model = deepcopy(model_best)
            scores = validation_model.validate(kfolds=10)
            print('\n  Cross Validation (K-fold, K=10) Metrics: ')
            if config['verbosity'] > 2:
                print('\n   Times: ')
                for key, value in scores['times'].items():
                    print('     %s: %.8f' % (key, value.mean()))
                    print('     ', value)
            print('\n   Scores: ')
            for key, value in scores['scores'].items():
                if config['verbosity'] > 1:
                    print('     %s: %.8f' % (value['description'],
                                             value['value'].mean()))
                    print('     ', value['value'])
                else:
                    print('     %s: %.8f' % (value['description'],
                                             value['value'].mean()))
            endtime = time.time()
            print('  Execution time = %.2f seconds' % (endtime - starttime))
            model_best.validation = scores
            print('\n***** Cross Validation Done! *****\n')
    if 'measuresfraction' in config.keys():
        model_best.measuresfraction = config['measuresfraction']
        model_best.measuresfraction_points = x_sample_train
    print('\n\n***** ALL DONE! *****\n')
    fn = model_best.savedata(parsec_exec.config, ' '.join(sys.argv))
    print('Model data saved on filename: %s' % fn)
    if modelFun:
        return {
            'config_selected': x_sample_train,
            'exec_times': y_sample_train,
            'exec_time_f_min_core_min': exec_time_f_min_core_min,
            'max_exec_time': max_exec_time,
            'min_exec_time': min_exec_time,
            'x_sample_test': x_sample_test,
            'y_sample_test': y_sample_test,
            'f_min': f_min,
            'best_parameter': model_best.sol,
            'error': model_best.error,
            'percentual_error': model_best.errorrel
        }
    toReturn = {
        'config_selected': x_sample_train,
        'exec_times': y_sample_train,
        'times': data_detach(times)['y'],
        'exec_time_f_min_core_min': exec_time_f_min_core_min,
        'max_exec_time': max_exec_time,
        'min_exec_time': min_exec_time,
        'x_sample_test': x_sample_test,
        'y_sample_test': y_sample_test,
        'f_min': f_min,
        'best_parameter': model_best.sol,
        'error': model_best.error,
        'percentual_error': model_best.errorrel
    }
    toReturn['x_sample_test'] = m_data['x']
    toReturn['y_sample_test'] = m_data['y']
    return toReturn
    # return {
    #     'best_parameter': model_best.sol,
    #     'error': model_best.error,
    #     'percentual_error': model_best.errorrel
    # }


# if __name__ == '__main__':
#     main()
