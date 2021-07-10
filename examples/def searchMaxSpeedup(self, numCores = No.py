def searchMaxSpeedup(self, numCores = None):

        if numCores!=None:
            cores = self.num_cores.split(',')
            freqs = self.freq_avail.split(',')
            bestSpeedup = None

            bestFreq=None
            bestCoreSpeedup=[-math.inf,-1,-1]
            core=numCores

            for freq in freqs:
                # Exp2: Calcular la mejor speedup sin limitaciones
                speedup = self.func_speedup(self.params, freq, core)
                if bestSpeedup == None:
                    bestSpeedup = [speedup, core, freq]
                elif bestSpeedup[0] < speedup:
                    bestSpeedup = [speedup, core, freq]

                
                # Exp1.1: Calcular la mejor frecuencia según el número de nucleos que de el mejor speedup

                
                if bestCoreSpeedup[0]<speedup:
                    bestCoreSpeedup= [speedup, core, freq]
            # Exp1.2: Verificar si la frecuencia elegida es la optima 
            # Comparar con los speedups reales
            returned=[]
            if numCores !=None:
                fr=bestCoreSpeedup[2]
                cr=bestCoreSpeedup[1]
                index=self.getIndexByCoreFreq(cr, fr)
                y_sample=self.y_sample_test[index]
                print(y_sample)
                print()
                print(bestCoreSpeedup[0])
                error = mean_squared_error([y_sample], [bestCoreSpeedup[0]])
                print("error : %.4f" % error)
                print("Best frequency: " + str(bestCoreSpeedup[2]) + "  for " + str(bestCoreSpeedup[1]) + "  cores")
                returned=bestCoreSpeedup

                return returned


        if not hasattr(self, 'x_sample_test'):
            self.params = self.getPSOParams()
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

                
        
        


        print("Cores: " + str(bestSpeedup[1]))
        print("Freq: " + str(bestSpeedup[2]))
        print("Speedup: " + str(bestSpeedup[0]))
        return returned