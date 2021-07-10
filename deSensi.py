from copy import deepcopy
import json
from dataprocess import ParsecLogsData
from math import ceil, sqrt

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import runmodel
from cpufreq import cpuFreq
import multiprocessing
import os
import time

import csv

import subprocess

# ts = training_size en runmodel
# measures = configuraciones ( frequencia y cores )


# alpha = 0.1
# leakage (I_leak)= 0.5nA
# Load capacitance (c) roughly proportional to the chip area = 14
# v I_leak + alpha * c * v*v * freq * core
# v = 19?
# 3.95
# 

class TestDeSensi:
    def __init__(self, package='parsec.blackscholes'):
        self.repetitions = 5
        self.setPackage(package)
        self.main_path = "/media/israel/ExtraSpaceLinux/proyectosI/memoryWall/"
        self.getFrequenciesCores()


    """
    taf = T min + ((T max − T min ) ∗ i)
    """
    def getTaf(self, i):
        return self.min_exec_time + ((self.max_exec_time - self.min_exec_time) * i)

    """
    The ideal algorithm has the knowledge of
    all the power consumption and execution times values in all the 
    possible conﬁgurations, so it is able to always choose the
    optimal conﬁguration.
    """
    def idealFunction(self, taf, best_time):
        best_ideal_time = [0, 0, np.inf, np.inf]
        for index, exec_time in enumerate(self.y_sample_test):
            # Check if power of two with bit manipulation
            # Check if package is fluidanimate which works only with cores 
            # power of two
            core = int(self.x_sample_test[index][1])
            if self.package == 'parsec.fluidanimate' and not ( (core != 0) and (core & (core-1) == 0)):
                continue

            index_pc = self.getIndexByCoreFreqPC(self.x_sample_test[index][1], self.x_sample_test[index][0])
            if index_pc == -1:
                continue

            power = self.powerConsumption['power_consumption'][index_pc]

            if exec_time <= taf and power < best_ideal_time[3]:
                best_ideal_time = [self.x_sample_test[index][0], self.x_sample_test[index][1], exec_time, power]

        print("COMPARE")
        print("Prediction: ", end="")
        print(best_time)
        print("Ideal: ", end="")
        print(best_ideal_time)
        print()
        p_error = 100 * abs(best_time[2] - best_ideal_time[2]) / best_ideal_time[2]
        if best_time[2] <= taf:
            if best_ideal_time[0] == best_time[0] and best_ideal_time[1] == best_time[1]:
                return ['SUCCESS', p_error]
            else:
                # 100 * (foundTime - idealTime) / idealTime
                return ['LOSS', p_error]
        else:
            return ['MISS', p_error]


    """
    This is the conﬁguration that minimises
    the power consumption while terminating in
    a time less than taf
    """
    def minPower(self, taf, freqs, cores):
        """
        :taf: Max execution time
        :freqs: List of frequencies
        :cores: List of cores
        """
        # frequency, cores, exec_time, power_consumption
        print("TAF: " + str(taf))
        # frecuencia, cores, tiempo_exec, energia
        best_time = [0, 0, np.inf, np.inf]
        for core in cores:
            # Check if power of two with bit manipulation
            # Check if package is fluidanimate which works only with cores 
            # power of two
            c = int(core)
            if self.package == 'parsec.fluidanimate' and not ( (c != 0) and (c & (c-1) == 0)):
                continue
            for freq in freqs:
                index = self.getIndexByCoreFreqPC(core, freq)
                if index == -1:
                    continue
                
                exec_time_prediction = self.performancePrediction(freq, core)
                power = self.powerConsumption['power_consumption'][index]
                # print("VOLT: " + str( power / (1.602176e-19 * exec_time_prediction)))
                # if self.chooseClosest(taf, exec_time_prediction, best_time, power):
                if exec_time_prediction <= taf and power < best_time[3]:
                    best_time = [int(freq), int(core), exec_time_prediction, power]

        return self.idealFunction(taf, best_time)
    

    def searchBestConfiguration(self):
        self.runModelPrediction()

        results = []
        for i in range(1,11):
            taf = self.getTaf(i/10)
            cores = self.num_cores.split(',')
            freqs = self.freq_avail.split(',')
            result = self.minPower(taf, freqs, cores)
            results.append([i/10, result])
        
        for r in results:
            print("I: " + str(r[0]))
            print(r[1][0] + ': %.4f' % r[1][1] + ' %')
            print()

    # values [B]
    def performancePrediction(self, freq, cores):
        """
        Model function to calculate the speedup without overhead.
        :param B1: T(1, f_min) * B
        :param B2: T(1, f_min) * (1 - B)
        :return: calculated speedup value
        :return: B1 * f_min / freq + B2 * f_min / (freq * cores)
        """
        freq = float(freq)
        cores = float(cores)
        return self.B1 * self.f_min / freq + self.B2 * self.f_min / (freq * cores)

    def isInConfig(self, core, freq):
        for config in self.config_selected:
            if str(core) == str(config[1]) and str(freq) == str(config[0]):
                return True

        return False

    def getIndexByCoreFreqPC(self, core, freq):
        for index, config in enumerate(self.powerConsumption['configuration']):
            if str(core) == str(config[1]) and str(freq) == str(config[0]):
                return index
        return -1


    def getIndexByCoreFreq(self, core, freq):
        for index, config in enumerate(self.x_sample_test):
            if str(core) == str(config[1]) and str(freq) == str(config[0]):
                return index

    def holdout(self):
        # Get 1% of the configurations
        """
        We select a certain percentage of the total conﬁgurations
        as input for the multiple linear regression
        algorithm, thus obtaining a model of the execution
        time and power consumption.
        """
        self.runModelPrediction()
        """
        After that, we use this model to predict the behaviour
        of the application in all the conﬁgurations not selected
        at the previous step.
        """
        cores = self.num_cores.split(',')
        freqs = self.freq_avail.split(',')
        error = 0
        for core in cores:
            for freq in freqs:
                if not self.isInConfig(core, freq):
                    exec_time_prediction = self.performancePrediction(freq, core)
                    """
                    Eventually, we compute the error of the predic-
                    tions by using the Mean Absolute Percentage Error
                    (MAPE), deﬁned as: e = 1/p sum(abs((A_i - P_i)/ P_i))
                    where p is the number of configurations
                    A_i is the real execution time 
                    P_i is the predicted execution time
                    """
                    index = self.getIndexByCoreFreq(core, freq)
                    single_error = abs( (self.y_sample_test[index] - exec_time_prediction) / exec_time_prediction )
                    # print(single_error)
                    # print("Error: %.4f" % single_error)            
                    error += single_error
        
        error = error / self.measures

        """
        The average accuracy is then computed as 100 − ε.
        """
        avg_accuracy = 100 - error
        print("AVG accuracy: %.4f" % avg_accuracy)
        print()

    # Funcion Flow Chart Fig 7
    # Main
    def plotMeasures(self, directory='result', package='parsec.blackscholes'):
        if os.path.isdir(directory):
            for root, dirs, files in os.walk(directory):
                self.runfiles = [name for name in files if
                                 name.startswith(package)]

            best_particle = None
            file_count = -1
            for file in self.runfiles:
                file_count = file_count + 1
                """
                # File structure:
                config
                  # Execution info (Ejm: measurefraction(measures))
                  :param measuresfraction: 64
                data
                  # Parsec exec values and model prediction values
                  params: Best particle of PSO (f, m1, m2, k)
                  error: MSE
                  errorrel: MSPE - Mean Square Percentage Error
                  parsecdata: Saves parsec execution values
                    dims: Variables modified in parsec executions (frequency and cores)
                    attrs: Package and command used
                    data: Execution values by size of frequency and each of size of cores
                          using formula: Ts / Tp (serial_time / parallel_time)
                    coords: Doesn't matter much
                      frequency
                      cores
                """
                f = open(directory + '/' + file)
                datadict = json.load(f)

                # print("contect")
                # print(datadict.keys())
                # print("Data")
                # print(datadict['data'].keys())
                data = datadict['data']
                params = data['params']
                error = data['error']
                errorrel = data['errorrel']
                if best_particle == None:
                    best_particle = {}
                    best_particle['error'] = error
                    best_particle['params'] = data['params']
                elif float(error) < float(best_particle['error']):
                    best_particle['error'] = error
                    best_particle['params'] = data['params']

                parsecJson = data['parsecdata']
                modelJson = data['speedupmodel']

                # data, parsecdata data: los speedups
                parsecData = parsecJson['data']
                modelData = modelJson['data']

                modelCoordsJson = modelJson['coords']
                parsecCoordsJson = parsecJson['coords']

                dimsLabels = data['parsecdata']['dims']
                # print(dimsLabels)

                # El orden de las frecuencias de los speedups (FRECUENCIA)
                parsecCoordsData = parsecCoordsJson['frequency']['data']
                # El orden de las frecuencias de las predicciones
                modelCoordsData = modelCoordsJson['frequency']['data']
                # modelCoordsData = modelCoordsJson['frequency']['data']

                # Usually parsecCoordsData has the correct order
                # Sorting parsec execution info
                # Sort execution values based on the considered order of frequencies
                parsecData = [x for _,x in sorted(zip(parsecCoordsData,parsecData))]
                parsecCoordsData.sort()
                # Sorting model execution info
                # Sort model prediction values based on the considered order of frequencies
                modelData = [x for _,x in sorted(zip(modelCoordsData, modelData))]
                modelCoordsData.sort()
                
                modelCoordsDataNew = modelCoordsData
                modelCoordsData = []

                parsecCoordsDataNew = parsecCoordsData
                parsecCoordsData = []

                parsecDataNew = parsecData
                parsecData = []

                for i in range(len(modelCoordsDataNew)):
                    if i%2 == 0:
                        parsecCoordsData.append(parsecCoordsDataNew[i] / 1000000)
                        modelCoordsData.append(modelCoordsDataNew[i] / 1000000)
                for idx in range(len(parsecDataNew)):
                    if idx%2 == 0:
                        parsecData.append(parsecDataNew[idx])
                    
                if file_count == len(self.runfiles) -1:
                    # X data: Frequency: modelCoordsData
                    # Y data: Cores: modelCoordsJson['cores']['data']
                    # Z data: Speedup: modelData
                    xModelData = []
                    yModelData = []
                    zModelData = []
                    
                    xParsecData = []
                    yParsecData = []
                    zParsecData = []

                    # Model data
                    for i in range(len(modelCoordsJson['cores']['data'])):
                        # 0.8,1.0,1.2....
                        # [modelCoordsData[i] for x in range(len(modelCoordsJson['cores']['data']))]
                        # 0.8,0.8,0.8,0.8
                        xModelData = xModelData + [modelCoordsData[i] for x in range(len(modelCoordsJson['cores']['data']))]

                    for i in range(len(modelCoordsJson['cores']['data'])):
                        yModelData = yModelData + modelCoordsJson['cores']['data']
                    
                    for idx in range(len(modelData)):
                        if idx%2 == 0:
                            zModelData = zModelData + modelData[idx]
                    
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    
                    yParsecData = parsecCoordsJson['cores']['data']
                    # print(len(parsecData))
                    for idx in range(len(parsecData)):
                        xParsecData = [parsecCoordsData[idx] for x in range(len(parsecCoordsJson['cores']['data']))]
                        zParsecData = parsecData[idx]
                        if idx == 0:
                            ax.plot3D(xParsecData, yParsecData, zParsecData, c='blue', linewidth=0.5, marker='x', label='Model')
                        else:
                            ax.plot3D(xParsecData, yParsecData, zParsecData, c='blue', linewidth=0.5, marker='x')
                        ax.scatter3D(xModelData, yModelData, zModelData, c='black', marker='o')
                        

                    ax.legend(loc="best")
                    ax.set_title(package + '\n Speedup model')
                    ax.set_xlabel('Frequency')
                    # cb = plt.colorbar(scat_plot, pad=0.2)
                    ax.set_ylabel('Number of cores')
                    ax.set_zlabel('Speedup')

                    plt.show()

                f.close()
            parameter_values = best_particle['params']
            parameter_values = parameter_values.split(',')
            parameter_values[0] = parameter_values[0][1:]
            parameter_values[3] = parameter_values[3][:len(parameter_values[3])-1]
            print(parameter_values)
            print(parameter_values[0])
            print(parameter_values[1])
            print(parameter_values[2])
            print(parameter_values[3])
            print("Best particle")
            print("Error: %.4f " % float(best_particle['error']))
            print("f: %.4f" % float(parameter_values[0]))
            print("k: %.4f" % float(parameter_values[3]))
            print("m1: %.4f" % float(parameter_values[1]))
            print("m2 %.4f" % float(parameter_values[2]))

    def runModelPrediction(self, measures = None):
        # self.parsec_runproccess('parsec.blackscholes', 'native', 1)       # run all parsec and return json with execution time
        self.ts = 4
        config_json = self.create_json()

        result = self.parsec_runmodel(config_json)

        self.B = float(result['best_parameter'][0])
        self.f_min = float(result['f_min'])
        self.exec_time_f_min_core_min = float(result['exec_time_f_min_core_min'])
        self.min_exec_time = float(result['min_exec_time'])
        self.max_exec_time = float(result['max_exec_time'])
        self.config_selected = result['config_selected']
        self.y_sample_test = result['y_sample_test']
        self.x_sample_test = result['x_sample_test']

        # print(self.y_sample_test)
        if self.package == 'parsec.fluidanimate':
            self.min_exec_time = np.inf
            self.max_exec_time = -1
            for index, exec_time in enumerate(self.y_sample_test):
                # Check if power of two with bit manipulation
                # Check if package is fluidanimate which works only with cores 
                # power of two
                core = int(self.x_sample_test[index][1])
                if (core != 0) and (core & (core-1) == 0):
                    if exec_time < self.min_exec_time:
                        self.min_exec_time = exec_time
                    elif exec_time > self.max_exec_time:
                        self.max_exec_time = exec_time
                
                

        self.B1 = self.exec_time_f_min_core_min * self.B
        self.B2 = self.exec_time_f_min_core_min * (1 - self.B)

        self.loadPowerConsumption()
        
    def loadPowerConsumption(self):
        self.powerConsumption = {'configuration': [], 'power_consumption': []}

        # powerConsumption Example: 1624239760.77281; execution-1-2200000-1; 116.3971996307373; 1243352163.0
        powerConsumption = []
        # Open power consumption file and put it in powerConsumption (local)
        with open('execution.' + self.package +'.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for _row in spamreader:
                row = _row[0].split(';')
                if row[1][:9] == 'execution':
                    powerConsumption.append(row)
                    
        # Get power consumption by configuration                   
        for index_pc in range(0, len(powerConsumption),3):
            values = []
            # Get values to compute median
            for pc in powerConsumption[index_pc:index_pc+3]:
                values.append(float(pc[3]))

            power_consumption = np.median(values)

            # Get configuration data (frequency - cores)
            # Example: execution-1-2200000-1
            config_data = powerConsumption[index_pc][1].split('-')
            pc_config = [int(config_data[2]), int(config_data[1])]
            if not pc_config in self.x_sample_test:
                continue

            power_consumption_ROI = -1

            for index_st, config in enumerate(self.x_sample_test):
                # Compare power consumption config here and runmodel config
                # Reason: power consumption config has overall time
                #         and runmodel config has ROI time
                # print(str(config[0]) +'=='+ str(int(pc_config[0])) +' and '+ str(config[1]) +'=='+ str(int(pc_config[1])))
                if str(config[0]) == str(int(pc_config[0])) and str(config[1]) == str(int(pc_config[1])):
                    # Compute rule of three to get power consumption in ROI
                    # power_consumption_ROI = (ROI_exec_time * power_consumption) / total_exec_time
                    power_consumption_ROI = (float(self.y_sample_test[index_st]) * (power_consumption)) / float(powerConsumption[index_pc][2])
                    break
            if power_consumption_ROI != -1:
                self.powerConsumption['configuration'].append(pc_config)
                self.powerConsumption['power_consumption'].append(power_consumption_ROI)

    # Funcion para llamar a runmodel con las frecuencias y cores que tenemos
    def parsec_runmodel(self, config_json):
        return runmodel.main(config_json, True)

    def setPackage(self, package):
        self.package = package

    def getFrequenciesCores(self):
        cf = cpuFreq()
        self.freq_avail = ''
        self.num_cores = ''
        self.measures = multiprocessing.cpu_count() * len(cf.available_frequencies)
        self.f_min = cf.available_frequencies[len(cf.available_frequencies) - 1]

        for i in range(0,multiprocessing.cpu_count()):
            self.num_cores += str(i + 1) +','
        self.num_cores = self.num_cores[:len(self.num_cores)-1]
        for freq in cf.available_frequencies:
            if freq < cf.available_frequencies[1]:
                self.freq_avail += str(freq) + ','
        self.freq_avail = self.freq_avail[:len(self.freq_avail)-1]

    # Funcion para llamar a runprocess pasando por parametro el json
    def parsec_runproccess(self, input, repetitions):
        print("Running package '", self.package ,"' with input '", input, "' with ",repetitions, " repetitions")

        print("Available frequencies: " + self.freq_avail)
        print("Available cores: " + self.num_cores)

        if "parsec.blackscholesNative.dat" in os.listdir(self.main_path):
            print("Result already available")
            return


        command = ['python3', 'runprocess.py', '-p', self.package, '-i', input, '-f', self.freq_avail, '-r', str(repetitions), self.num_cores]
        res = subprocess.Popen(command, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        while res.poll() == None:
            continue

        print("package done")

    # Funcion para crear el json
    def create_json(self):
        # values [B]
        # repetitions ejecuta n veces el PSO
        # maxiter ejecuta cada iteracion dentro del PSO n veces
        if self.package != 'parsec.fluidanimate':
            return {
            "algorithm": "pso",
            "parsecpydatafilepath": self.main_path + self.package + "Native.dat",
            "resultsfolder": "./resultDeSensi",
            "modelcodefilepath": self.main_path + "examples/modelfunc_deSensi.py",
            "size": 200,
            "repetitions": self.repetitions,
            "w": 0.8,
            "c1": 1,
            "c2": 4,
            "maxiter": 100,
            "lowervalues": [0],
            "uppervalues": [1],
            "threads": 4,
            "measuresfraction": self.ts,
            "crossvalidation": False,
            "verbosity": 1
            }
        else:
            return {
            "algorithm": "pso",
            "parsecpydatafilepath": self.main_path + self.package + "Native.dat",
            "resultsfolder": "./resultDeSensi",
            "modelcodefilepath": self.main_path + "examples/modelfunc_deSensi.py",
            "size": 200,
            "repetitions": self.repetitions,
            "w": 0.8,
            "c1": 1,
            "c2": 4,
            "maxiter": 100,
            "lowervalues": [0],
            "uppervalues": [1],
            "threads": 4,
            "measuresfraction": self.ts,
            "crossvalidation": False,
            "verbosity": 1,
            "powerOfTwo": True
            }

test = TestDeSensi()

test.setPackage("parsec.fluidanimate")
# starttime = time.time()
# test.holdout()
# endtime = time.time()
# print('De sensi execution time = %.2f seconds' % (endtime - starttime))

starttime = time.time()
test.searchBestConfiguration()
endtime = time.time()
print('De sensi execution time = %.2f seconds' % (endtime - starttime))


# starttime = time.time()
# test.plotMeasures()
# print('plot measures execution time = %.2f seconds' % (endtime - starttime))
# endtime = time.time()
# test.main()