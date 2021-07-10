from copy import deepcopy
import json
from posixpath import split
import sys
from dataprocess import ParsecLogsData
from math import sqrt

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
from sklearn.metrics import mean_squared_error
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import runmodel
from cpufreq import cpuFreq
import multiprocessing
import os
import time
import math

import csv

import subprocess

# ts = training_size en runmodel
# measures = configuraciones ( frequencia y cores )


class TestParsecpy:
    def __init__(self, package='parsec.blackscholes'):
        self.setPackage(package)
        self.main_path = "/media/israel/ExtraSpaceLinux/proyectosI/memoryWall/"
        self.repetitions = 5
        self.getFrequenciesCores()

    def calculateMu(self, p, m1, m2):
        return min(m1 + m2 / p, 1)

    def calculateRho(self, k, phi):
        return 1 + k * phi

    # values [f, m1, m2, k]
    def func_speedup(self, param, freq, cores):
        """
        Model function to calculate the speedup without overhead.

        :param fparam: Actual parameters values
        :param p: Numbers of cores used on model data
        :param n: Problems size used on model data
        :return: calculated speedup value
        """
        freq = float(freq)
        cores = float(cores)
        
        phi = (freq / 1e6) / 2.6

        #(1 − mu) + rho mu
        up_arg=(1-self.calculateMu(1,param[1],param[2])) + self.calculateRho(param[3],phi) * self.calculateMu(1,param[1],param[2])

        # ((1 − mu_p ) + rho mu_p ) * (1 − f ) + f/p)
        leftSide = ((1- self.calculateMu(cores, param[1], param[2])) + self.calculateRho(param[3], phi)) * ((1-param[0])+ param[0]/cores)

        # rho mu_p
        rightSide = phi * self.calculateMu(cores, param[1], param[2])

        #Calculate max btw below args
        below_arg = max (leftSide, rightSide)

        speedup = up_arg / below_arg
        return speedup


    def getPSOParams(self):
        self.ts = 4
        config_json = self.create_json()

        result =  self.parsec_runmodel(config_json)
        self.params = result['best_parameter']
        self.min_speedup = float(result['min_exec_time'])
        self.max_speedup = float(result['max_exec_time'])
        self.y_sample_test = result['y_sample_test']
        self.x_sample_test = result['x_sample_test']
        self.times = result['times']

        if self.package == 'parsec.fluidanimate':
            self.min_speedup = np.inf
            self.max_speedup = -1
            for index, exec_time in enumerate(self.y_sample_test):
                # Check if power of two with bit manipulation
                # Check if package is fluidanimate which works only with cores 
                # power of two
                core = int(self.x_sample_test[index][1])
                if (core != 0) and (core & (core-1) == 0):
                    if exec_time < self.min_speedup:
                        self.min_speedup = exec_time
                    elif exec_time > self.max_speedup:
                        self.max_speedup = exec_time
                
                
        
        self.loadPowerConsumption()

        return result['best_parameter']

    """
    SPEEDUP LIMITED BY POWER CONSUMPTION
    """

    """
    taf = T min + ((T max − T min ) ∗ i)
    """
    def getTaf(self, i):
        return self.min_speedup + ((self.max_speedup - self.min_speedup) * i)

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

    def chooseClosest(self, taf, exec_time_prediction, best_time, power):
        if exec_time_prediction <= taf and power < best_time[3]:
            return True
            
        return False

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

                speedup_prediction = self.func_speedup(self.params,freq, core)
                power = self.powerConsumption['power_consumption'][index]

                if self.chooseClosest(taf, speedup_prediction, best_time, power):
                    best_time = [int(freq), int(core), speedup_prediction, power]

        return self.idealFunction(taf, best_time)


    def searchBestConfiguration(self):
        self.getPSOParams()
        print("Configuration with lower power consumption")
        min_power = [0, np.inf]
        for index, power in enumerate(self.powerConsumption['power_consumption']):
            c = int(self.powerConsumption['configuration'][index][1])
            if self.package == 'parsec.fluidanimate' and not ( (c != 0) and (c & (c-1) == 0)):
                continue
            if power < min_power[1]:
                min_power = [index, power]
        print(min_power)
        print(self.powerConsumption['configuration'][min_power[0]])
        print()

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
            # if r[1] == 'SUCCESS' or r[1] == 'MISS':
            #     print("Test result:" + r[1])
            # else:
            #     print("Test result: %.4f" % r[1])

    def getIndexByCoreFreqPC(self, core, freq):
        for index, config in enumerate(self.powerConsumption['configuration']):
            if str(core) == str(config[1]) and str(freq) == str(config[0]):
                return index
        return -1


    def getIndexByCoreFreq(self, core, freq):
        for index, config in enumerate(self.x_sample_test):
            if str(core) == str(config[1]) and str(freq) == str(config[0]):
                return index

    def plot3DBar(self, plotType = 2):
        if not hasattr(self, 'x_sample_test'):
            self.getPSOParams()
            
        fig = plt.figure()
        
        if plotType == 2:
            ax = fig.add_subplot(211, projection='3d')
        else:
            ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Number of cores')

        if plotType == 0: # Plot power consumption
            ax.set_title(self.package + '\n Power consumption')
            ax.set_zlabel('Joules (GJ)')
            barData = self.plotPowerConsumption()
        elif plotType == 1: # plot ROI execution time
            ax.set_title(self.package + '\n Execution times')
            ax.set_zlabel('Execution time (min)')
            barData = self.plotROIExecTime()
        elif plotType == 2: # plot comparing both
            barData = self.plotPowerConsumption()
            barData2 = self.plotROIExecTime()
            ax2 = fig.add_subplot(221, projection='3d')
            ax2.set_xlabel('Frequency (GHz)')
            ax2.set_ylabel('Number of cores')
            ax.set_zlabel('Joules (GJ)')
            ax2.set_zlabel('Execution time (min)')
            ax2.bar3d(barData2[0], barData2[1], barData2[2], barData2[3], barData2[4], barData2[5], color='#00ceaa', shade=True)
        
        ax.bar3d(barData[0], barData[1], barData[2], barData[3], barData[4], barData[5], shade=True)

        plt.show()

    def plotPowerConsumption(self):
        self.loadPowerConsumption()
        """
        x Axis: Frequency
        y Axis: Cores
        z Axis: Power Consumption
        """ 
        xpos = []
        ypos = []
        dz = []
        configs = self.powerConsumption['configuration']
        powerConsumption = self.powerConsumption['power_consumption']
        for index, config in enumerate(configs):
            freqMod = config[0] / 1e6
            powerMod = powerConsumption[index] / 1e9
            xpos.append(freqMod)
            ypos.append(config[1])
            dz.append(powerMod)

        num_elements = len(xpos)
        zpos = np.zeros(num_elements)
        dx = .05 * np.ones(num_elements)
        dy = .25 * np.ones(num_elements)
        return [xpos, ypos, zpos, dx, dy, dz]
    
    def plotROIExecTime(self):
        if not hasattr(self, 'x_sample_test'):
            self.getPSOParams()
        """
        x Axis: Frequency
        y Axis: Cores
        z Axis: Power Consumption
        """ 
        xpos = []
        ypos = []
        dz = []

        for index, config in enumerate(self.x_sample_test):
            freqMod = float(config[0]) / 1e6
            timeMod = self.times[index] / 60
            xpos.append(freqMod)
            ypos.append(config[1])
            dz.append(timeMod)

        num_elements = len(xpos)
        zpos = np.zeros(num_elements)
        dx = .05 * np.ones(num_elements)
        dy = .25 * np.ones(num_elements)
        return [xpos, ypos, zpos, dx, dy, dz]

    def loadPowerConsumption(self):
        if not hasattr(self, 'x_sample_test'):
            self.getPSOParams()
            return
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
        for index_pc in range(0,len(powerConsumption),3):
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
                if str(config[0]) == str(int(pc_config[0])) and str(config[1]) == str(int(pc_config[1])):
                    # Compute rule of three to get power consumption in ROI
                    # power_consumption_ROI = (ROI_exec_time * power_consumption) / total_exec_time
                    power_consumption_ROI = (float(self.times[index_st]) * (power_consumption)) / float(powerConsumption[index_pc][2])
                    break
            if power_consumption_ROI != -1:
                self.powerConsumption['configuration'].append(pc_config)
                self.powerConsumption['power_consumption'].append(power_consumption_ROI)

    """
    END SPEEDUP LIMITED BY POWER CONSUMPTION
    """

    """
    Nosotros hallamos el c y f ( la mejor configuración que maximize el speedup) 
    según los parámetros que devuelva PSO. Buscar en c*f calculando el modelo
    speedupReal = 3.2

    :params: [f, m1, m2, k]
    """
    def searchMaxSpeedup(self, numCores = None):
        cores = self.num_cores.split(',')
        freqs = self.freq_avail.split(',')
        bestSpeedup = None

        bestFreq=None
        bestCoreSpeedup=[-math.inf,-1,-1]
        for core in cores:
            for freq in freqs:
                # Exp2: Calcular la mejor speedup sin limitaciones
                speedup = self.func_speedup(self.params, freq, core)
                if bestSpeedup == None:
                    bestSpeedup = [speedup, core, freq]
                elif bestSpeedup[0] < speedup:
                    bestSpeedup = [speedup, core, freq]

                
                 # Exp1.1: Calcular la mejor frecuencia según el número de nucleos que de el mejor speedup

                if numCores != None and int(core) == numCores:
                    if bestCoreSpeedup[0] < speedup:
                        bestCoreSpeedup = [speedup, core, freq]
        # Exp1.2: Verificar si la frecuencia elegida es la optima 
        # Comparar con los speedups reales
        returned=[]
        if numCores !=None:
            fr=bestCoreSpeedup[2]
            cr=bestCoreSpeedup[1]
            # Para cr, buscar cada frecuencia su speedup real y devolver el indice del mayor speeduop
            # Ejm: 2.1 cr

            #cores = self.num_cores.split(',')
            freqs = self.freq_avail.split(',')

            # for index, core in enumerate(cores):
            best_real_speedup=-math.inf
            best_freq=None
            for freq in freqs:
                index=self.getIndexByCoreFreq(cr, freq)
                y_sample=self.y_sample_test[index]
                if(y_sample>best_real_speedup):
                    best_real_speedup=y_sample
                    best_freq=freq

            if(int(best_freq)==int(fr) and int(numCores)==int(cr)):
                print("SUCCESS")
            else:
                error = mean_squared_error([best_real_speedup], [bestCoreSpeedup[0]])
                error = 100* (((bestCoreSpeedup[0] - best_real_speedup) / best_real_speedup)**2)
                if error<=0.01:
                    print("ACCEPTABLE")
                    print("Error less than 1% ")
                else:
                    print("MISS")
                
                print("error : %.4f" % error)
                print("Best frequency: " + str(best_freq) + "  for " + str(numCores) + "  cores:  speedup " + str(best_real_speedup))
                print("Best found frequency: " + str(bestCoreSpeedup[2]) + "  for " + str(bestCoreSpeedup[1]) + "  cores: speedup " + str(bestCoreSpeedup[0]))
                print()
                
            returned=bestCoreSpeedup

        # print("Cores: " + str(bestSpeedup[1]))
        # print("Freq: " + str(bestSpeedup[2]))
        # print("Speedup: " + str(bestSpeedup[0]))
        return returned
    
    def eachCoresFreq(self):
        self.params = self.getPSOParams()

        x=1
        if(self.package == 'parsec.fluidanimate'):
            x = 2
        dictionary={}
        while(x<=multiprocessing.cpu_count()):
            val=self.searchMaxSpeedup(x)
            dictionary[str(x)]=val[2]
            if(self.package == 'parsec.fluidanimate'):
                x = x*2
            else:
                x+=1

        print(dictionary)

        return dictionary
            
    def runMultipleModels(self):
        modelos=3
        # Plot data
        plt.title('app = ' + self.package)
        plt.xlabel('Train size')
        plt.ylabel('Median of errors')
        yTicks = []
        allMSE = []
        allSD = []
        for index in range(0, modelos):
            #print(json_arr[index])
            tsList, median, label, standard_deviation  = self.runModelPrediction(index)
            allMSE.append(median)
            allSD.append(standard_deviation)
            
            yTicks.extend(median)
            plt.plot(tsList, median, label=label)
        
        print("ALLMSE")
        print(allMSE)
        print()
        print("ALL SD")
        print(allSD)
        
        plt.xticks(tsList)
        plt.yticks([1, 0.1, 0.01, 0.001])
        plt.legend()
        plt.show()

    def runPlotMeasures(self):
        modelos = 3
        for index in range(modelos):
            if index == 0:
                directory = 'result' + self.package
            elif index == 1:
                directory = 'resultSVR' + self.package
            elif index == 2:
                directory = 'resultAmdahlPSO' + self.package
            self.plotMeasures(directory)

    def sortData(self, data):
        # data, parsecdata data: los speedups
        parsecData = data['parsecdata']['data']
        modelData = data['speedupmodel']['data']

        # El orden de las frecuencias de los speedups (FRECUENCIA)
        parsecCoordsData = data['parsecdata']['coords']['frequency']['data']

        # El orden de las frecuencias de las predicciones
        modelCoordsData = data['speedupmodel']['coords']['frequency']['data']
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
        modelDataNew = modelData
        modelData = []

        for i in range(len(modelCoordsDataNew)):
            if i%2 == 0:
                parsecCoordsData.append(parsecCoordsDataNew[i] / 1000000)
                modelCoordsData.append(modelCoordsDataNew[i] / 1000000)
        for idx in range(len(parsecDataNew)):
            if idx%2 == 0:
                parsecData.append(parsecDataNew[idx])
                modelData.append(modelDataNew[idx])
        
        return parsecData, parsecCoordsData, modelData, modelCoordsData

    # Funcion Flow Chart Fig 7
    # Main
    def plotMeasures(self, directory='result'):
        if not os.path.isdir(directory):
            return 

        for root, dirs, files in os.walk(directory):
            self.runfiles = [name for name in files if
                                name.startswith(self.package)]

        best_particle = {'error': np.inf}
        for file in self.runfiles:
            """
            # File structure:
            config
                # Execution info (Ejm: measurefraction(measures))
                measuresfraction: 64
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

            data = datadict['data']
            error = data['error']

            # chooses for best particle
            if float(error) < float(best_particle['error']):
                best_particle['error'] = error
                if 'params' in data.keys():
                    best_particle['params'] = data['params']

                # gets and sorts model and parsec data
                parsecData, parsecCoordsData, modelData, modelCoordsData = self.sortData(data)
                best_particle['modelData'] = modelData
                best_particle['parsecData'] = parsecData
                best_particle['parsecCoordsData'] = parsecCoordsData
                best_particle['modelCoordsData'] = modelCoordsData

            f.close()

        # X data: Frequency: modelCoordsData
        # Y data: Cores: modelCoordsJson['cores']['data']
        # Z data: Speedup: modelData
        xModelData = []
        yModelData = []
        zModelData = []
        
        xParsecData = []
        yParsecData = []
        zParsecData = []

        cores = self.num_cores.split(",")
        cores = [int(core) for core in cores]
        cores = cores[1:]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        yParsecData = cores
        yModelData = cores
        for idx in range(len(parsecData)):
            # Load Parsec data
            xParsecData = [parsecCoordsData[idx] for x in range(len(parsecCoordsData))]
            zParsecData = parsecData[idx][:]

            xModelData = [modelCoordsData[idx] for x in range(len(modelCoordsData))]
            zModelData = modelData[idx][:]

            if idx == 0:
                # print(xParsecData)
                # print(yParsecData)
                # print(zParsecData)
                
                ax.scatter3D(xParsecData, yParsecData, zParsecData, c='blue', linewidth=0.5, marker='o', label='Real measurements')
                ax.plot3D(xModelData, yModelData, zModelData, c='black', marker='x',  label='Model')
            else:
                ax.scatter3D(xParsecData, yParsecData, zParsecData, c='blue', linewidth=0.5, marker='o')
                ax.plot3D(xModelData, yModelData, zModelData, c='black', marker='x')
            
        ax.legend(loc="best")
        ax.set_title(self.package + '\n Speedup model')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Number of cores')
        ax.set_zlabel('Speedup')
        plt.show()

        # if 'params' in best_particle.keys():
        #     parameter_values = best_particle['params']
        #     parameter_values = parameter_values.split(',')
        #     parameter_values[0] = parameter_values[0][1:]
        #     parameter_values[3] = parameter_values[3][:len(parameter_values[3])-1]
        print("Best particle")
        print("Error: %.4f " % float(best_particle['error']))

    def runModelPrediction(self, index):
        self.ts = 4
        # self.parsec_runproccess('parsec.blackscholes', 'native', 1)       # run all parsec and return json with execution time
        median = []
        standard_deviation = []
        tsList = []
        best_particle = {}
        best_particle['error'] = 1 * np.inf
        while self.ts < self.measures:
            MSEs = []
            if(index==0):
                config_json=self.create_json()
            if (index==1):
                config_json=self.create_jsonSVR()
            elif (index==2):
                config_json=self.create_jsonAmdahl()
            for i in range(0, 100):
                # Llamar parsec_runmodel:
                # 1. Para cada modelo: Ajustar los parametros usando PSO
                # 2. Para cada modelo predice speedup y calcular MSE
                # 3. Para cada modelo guarda el MSE
                result = self.parsec_runmodel(config_json)
                MSEs.append(result['error'])
                
                if float(result['error']) < float(best_particle['error']):
                    best_particle = deepcopy(result)

            # Por cada modelo calcular la media y la desviacion estandar de los MSE
            _median = np.sum(MSEs)/ len(MSEs)
            median.append(_median)

            # standard_deviation = np.std(MSEs)
            _standard_deviation = 0
            for measure in MSEs:
                _standard_deviation += (measure - _median)**2
            _standard_deviation /= len(MSEs)
            _standard_deviation = sqrt(_standard_deviation)
            standard_deviation.append(_standard_deviation)

            tsList.append(self.ts)
            self.ts = self.ts * 2

        print("Mejor configuracion encontrada")
        parameter_values = best_particle['best_parameter']
        print("MSE: %.4f" % float(best_particle['error']))
        print("Percentual error: %" + "%.4f" % float(best_particle['percentual_error']))
        if index == 0:
            print("f: %.4f" % float(parameter_values[0]))
            print("k: %.4f" % float(parameter_values[3]))
            print("m1: %.4f" % float(parameter_values[1]))
            print("m2 %.4f" % float(parameter_values[2]))
        print()

        if index == 0:
            algoName = "Proposed model"
        elif index == 1:
            algoName = "SVR"
        elif index == 2:
            algoName = "Amdahl"
        
        return tsList, median, algoName, standard_deviation 

    # Funcion para llamar a runmodel con las frecuencias y cores que tenemos
    def parsec_runmodel(self, config_json):
        return runmodel.main(config_json)

    def setPackage(self, package):
        self.package = package

    def getFrequenciesCores(self):
        cf = cpuFreq()
        self.freq_avail = ''
        self.num_cores = ''
        self.measures = multiprocessing.cpu_count() * len(cf.available_frequencies)
        for i in range(0,multiprocessing.cpu_count()):
            self.num_cores += str(i + 1) +','
        self.num_cores = self.num_cores[:len(self.num_cores)-1]
        for freq in cf.available_frequencies:
            if freq < cf.available_frequencies[1]:
                self.freq_avail += str(freq) + ','
        self.freq_avail = self.freq_avail[:len(self.freq_avail)-1]

    # Funcion para llamar a runprocess pasando por parametro el json
    def parsec_runproccess(self):
        self.getFrequenciesCores()
        print("Running package '", self.package ,"' with native input with ", str(self.repetitions), " repetitions")

        print("Available frequencies: " + self.freq_avail)
        print("Available cores: " + self.num_cores)

        if "parsec.blackscholesNative.dat" in os.listdir(self.main_path):
            print("Result already available")
            return


        command = ['python3', 'runprocess.py', '-p', self.package, '-i', input, '-f', self.freq_avail, '-r', str(self.repetitions), self.num_cores]
        res = subprocess.Popen(command, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        while res.poll() is None:
                continue

        print("package done")

    # Funcion para crear el json
    def create_json(self):
        # values [f, m1, m2, k]
        # repetitions ejecuta n veces el PSO
        # maxiter ejecuta cada iteracion dentro del PSO n veces
        if self.package != 'parsec.fluidanimate':
            return {
            "algorithm": "pso",
            "parsecpydatafilepath": self.main_path + self.package + "Native.dat",
            "resultsfolder": "./result" + self.package,
            "modelcodefilepath": self.main_path + "examples/modelfunc_example_inputsize_cores.py",
            "size": 200,
            "repetitions": self.repetitions,
            "w": 0.8,
            "c1": 1,
            "c2": 4,
            "maxiter": 100,
            "lowervalues": [ 0, 0, 0, 0 ],
            "uppervalues": [ 1, 1, 1, 10 ],
            "threads": 4,
            "measuresfraction": self.ts,
            "crossvalidation": False,
            "verbosity": 1
            }
        else:
            return {
            "algorithm": "pso",
            "parsecpydatafilepath": self.main_path + self.package + "Native.dat",
            "resultsfolder": "./result" + self.package,
            "modelcodefilepath": self.main_path + "examples/modelfunc_example_inputsize_cores.py",
            "size": 200,
            "repetitions": self.repetitions,
            "w": 0.8,
            "c1": 1,
            "c2": 4,
            "maxiter": 100,
            "lowervalues": [ 0, 0, 0, 0 ],
            "uppervalues": [ 1, 1, 1, 10 ],
            "threads": 4,
            "measuresfraction": self.ts,
            "crossvalidation": False,
            "verbosity": 1,
            "powerOfTwo": True
            }
    # Funcion para crear el json
    #SVR algorithm
    def create_jsonSVR(self):
        # values [f, m1, m2, k]
        # repetitions ejecuta n veces el PSO
        # maxiter ejecuta cada iteracion dentro del PSO n veces
        return {
        "algorithm": "svr",
        "parsecpydatafilepath": self.main_path + "parsec.blackscholesNative.dat",
        "resultsfolder": "./resultSVR"  + self.package,
        "modelcodefilepath": self.main_path + "examples/modelfunc_example_inputsize_cores.py",
        "crossvalidation-folds": 3,
        "c_grid": [100, 1000],
        "gamma_grid": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
        "repetitions": self.repetitions,
        "measuresfraction": self.ts,
        "crossvalidation": False,
        "verbosity": 1
        }

    #CSA algorithm
    def create_jsonCSA(self):
        # values [f, m1, m2, k]
        # repetitions ejecuta n veces el PSO
        # maxiter ejecuta cada iteracion dentro del PSO n veces
        return {
        "algorithm": "csa",
        "parsecpydatafilepath": self.main_path + "parsec.blackscholesNative.dat",
        "resultsfolder": "./result",
        "modelcodefilepath": self.main_path + "examples/modelfunc_example_inputsize_cores.py",
        "size": 200,
        "repetitions": self.repetitions,
        "w": 1,
        "c1": 1,
        "c2": 4,
        "maxiter": 100,
        "lowervalues": [ 0, 0, 0, 0 ],
        "uppervalues": [ 1, 1, 1, 10 ],
        "threads": 4,
        "measuresfraction": self.ts,
        "crossvalidation": False,
        "verbosity": 1
        }
    
    #CSA algorithm
    def create_jsonAmdahl(self):
        # values [f]
        # repetitions ejecuta n veces el PSO
        # maxiter ejecuta cada iteracion dentro del PSO n veces
        return {
        "algorithm": "pso",
        "parsecpydatafilepath": self.main_path + "parsec.blackscholesNative.dat",
        "resultsfolder": "./resultAmdahlPSO"  + self.package,
        "modelcodefilepath": self.main_path + "examples/modelfunc_Amdahl.py",
        "size": 200,
        "repetitions": self.repetitions,
        "w": 1,
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

test = TestParsecpy()

test.setPackage('parsec.blackscholes')
# test.plot3DBar(0)
# starttime = time.time()
# test.runMultipleModels()
# endtime = time.time()
# print('Execution time = %.2f seconds' % (endtime - starttime))
# print()

# test.plotMeasures('resultparsec.blackscholes')

print("Correr el aporte2: Encontrar la configuracion que minimice el consumo de energia")
starttime = time.time()
test.searchBestConfiguration()
endtime = time.time()
print('Execution time = %.2f seconds' % (endtime - starttime))
print()


# print("BLACKSCHOLES")
# #  Correr la grafica de comparacion de MSEs medias
# print("Correr la grafica de comparacion de MSEs medias")
# test.eachCoresFreq()

# starttime = time.time()
# test.runMultipleModels()
# endtime = time.time()
# print('Execution time = %.2f seconds' % (endtime - starttime))
# print()

# print("Correr la grafica 3D mostrando la precision de la prediccion speedup comparado con los reales")
# starttime = time.time()
# test.runPlotMeasures()
# endtime = time.time()
# print('Execution time = %.2f seconds' % (endtime - starttime))
# print()

# print("Correr el aporte1: Encontrar la mejor frecuencia segun el numero de nucleos")
# starttime = time.time()
# test.eachCoresFreq()
# endtime = time.time()
# print('Execution time = %.2f seconds' % (endtime - starttime))
# print()

# print("Correr el aporte2: Encontrar la configuracion que minimice el consumo de energia")
# starttime = time.time()
# test.searchBestConfiguration()
# endtime = time.time()
# print('Execution time = %.2f seconds' % (endtime - starttime))
# print()

# print("Crear la grafica de barras 3D para analizar el consumo de energia y speedup")
# test.plot3DBar()