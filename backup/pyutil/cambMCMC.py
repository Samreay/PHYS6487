from sam import *
import numpy as np
import os
import matplotlib, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import copy
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
import psutil
import time
import sys



class CambMCMC(object):
    """ Doco """
    
    def __init__(self, camb4pyLoadPath=None, debug=False, fast=False, sigmaFrac=250.0):
	
        self.debug = False   
        self.finalData = None
        self.finalDataGrid = None
        self.finalOutput = None
        self.finalOutputModel = None
    
        self.camb = None
        self.cambArgs = {}
        self.cambParams = []
        self.cambNumParams = 0
    
        self.gridPoints = 5
        self.gridActive = True
    
        self.mcmcActive = True
        self.mcmcPoints = 100
        self.mcmcBurnIn = 0.0
        self.mcmcThinning = 1
        self.mcmcNumWalks = 10
    
        self.saveFolder = None
        self.saveFrequencyGrid = 50
        self.saveFrequencyMCMC = 20
    
        self.dataX = None
        self.dataY = None
        self.dataE = None  
        self.covariance = None
        self.dataAutoCorrelation = True
        self.dataSource = 'transfer_matterpower'
        self.columnX = 'k/h'
        self.columnY = 'power'
        self.gridData = None
        self.mcmcData = []
        self.sigmaFrac = sigmaFrac
    
    
        self.gridDataFile = 'grid.txt'
        self.mcmcDataFile = 'mcmc%d.txt'    
    
        self.finalNumberOfPoints = 1e5
        self.finalMeshLength = None
	
        if camb4pyLoadPath is not None:
            self.camb = camb4py.load(executable=camb4pyLoadPath, debug=debug)
        else:
            self.camb = camb4py.load(debug=debug)
        if fast:
            self.cambArgs['do_lensing'] = False
            self.cambArgs['accurate_polarization'] = False
            self.cambArgs['high_accuracy_default'] = False
        self.debug = debug
		
    def setFinalNumberOfPoints(self, num=1e5):
        self.finalNumberOfPoints = num
        self._updateMeshLength()
        
    def _updateMeshLength(self):
        if self.cambNumParams == 0:
            self.finalMeshLength = None
        elif self.cambNumParams == 1:
            self.finalmeshLength = self.finalNumberOfPoints / 2
        else:
            self.finalMeshLength = np.floor(np.power((self.finalNumberOfPoints/(self.cambNumParams + 1)), 1.0 / (self.cambNumParams)))        
            for i in self.cambParams:
                i['grid2'] = np.linspace(i['min'], i['max'], self.finalMeshLength)
        self.finalDataGrid = None
        
    def setDefaults(self, **args):
        self.cambArgs = args
        
    def addVariable(self, name, minValue, maxValue, sigma=None):
        if sigma is None:
            sigma = (maxValue - minValue) / self.sigmaFrac
        self.cambParams.append({
            'name' : name,
            'min'  : minValue,
            'max'  : maxValue,
            'sigma': sigma,
            'grid' : np.linspace(minValue, maxValue, self.gridPoints),
        })
        self.cambNumParams = len(self.cambParams)
        self._updateMeshLength()
    
    def configureGridSearch(self, active=True, n=5):
        self.gridActive = active
        self.gridPoints = n
        for i in self.cambParams:
            i['grid'] = np.linspace(i['min'], i['max'], n)

        
    def configureMCMC(self, active=True, n=100, burnIn=0.0, thinning=1, numWalks=10):
        """ Configure the MCMC part of the search
        
        burnIn represents the fraction of initial points to discard
        thinning reduces autocorrelation by only accepting points i % thinning = 0
        numStarts indicates how many times to randomly generate a starting position and walk
        
        """
        self.mcmcActive = active
        self.mcmcPoints = n
        self.mcmcBurnIn = burnIn
        self.mcmcThinning = thinning
        self.mcmcNumWalks = numWalks
    
    def _get_actual_filename(self, name):
            dirs = name.split('\\')
            # disk letter
            test_name = [dirs[0].upper()]
            for d in dirs[1:]:
                test_name += ["%s[%s]" % (d[:-1], d[-1])]
            res = glob.glob('\\'.join(test_name))
            if not res:
                #File not found
                return None
            return res[0]    
    
    def configureSaveFolder(self, folder, saveFrequencyGrid=50, saveFrequencyMCMC=20):
        """ If specificied, saves results to a temporary folder.
        
        Makes CambMCMC interruptable as it will reload results. Save frequency
        determines to save after x amount of accepted data points
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.saveFolder = self._get_actual_filename(os.path.abspath(folder))
        self.saveFrequencyGrid = saveFrequencyGrid
        self.saveFrequencyMCMC = saveFrequencyMCMC
        
    def setPhysicalData(self, dataX, dataY, dataE, dataSource, columnX, columnY):
        """ Assumes transfer function of matter, with x being cl, y being power """      
        self.dataX = dataX
        self.dataY = dataY
        self.dataE = dataE
        self.dataSource = dataSource
        self.columnX = columnX
        self.columnY = columnY
        
    def setData(self, dataX, dataY, covariance, autocorrelation=True):
        self.dataX = dataX
        self.dataY = dataY
        self.covariance = covariance[:,:,3]
        self.dataE = np.zeros(self.dataX.size)
        for i in range(dataX.size):
            self.dataE[i] = np.sqrt(covariance[i,i,2])
        self.dataAutoCorrelation = autocorrelation
    
    def _getInitialGridData(self):
        self.gridData = np.zeros((self.gridPoints ** self.cambNumParams, self.cambNumParams + 1))
        
        self.gridData[:, -1] -= 1
        
        arrs = [i['grid'] for i in self.cambParams]
        
        for i in range(self.gridPoints ** self.cambNumParams):
            for j, spacing in enumerate(arrs):
                self.gridData[i,j] = spacing[(i / (self.gridPoints ** (self.cambNumParams - 1 - j))) % self.gridPoints]
        self._saveGridData()
        
    def _saveGridData(self):
        if self.saveFolder is not None:
            np.savetxt(self.saveFolder + os.sep + self.gridDataFile, self.gridData)
            self._debug("Saved Grid Data")
    
    def _saveMCMCData(self, index=None):
        if self.saveFolder is not None:        
            if index is None:
                for i in range(len(self.mcmcData)):
                    self._saveMCMCData(i)
            else:
                np.savetxt(self.saveFolder + os.sep + (self.mcmcDataFile % index), self.mcmcData[index][self.mcmcData[index][:,-2] != -1])
                self._debug("Saved MCMC %d Data" % index)
            
    
    def _loadGridData(self):
        if self.saveFolder is not None:
            try:
                self.gridData = np.loadtxt(self.saveFolder + os.sep + self.gridDataFile)
            except:
                self._getInitialGridData()
        else:
            self._getInitialGridData()
            
    def _getNewWalk(self):
        walk = np.zeros((self.mcmcPoints, self.cambNumParams + 2))
        walk[:, -2] -= 1
        return walk
        
    def _loadMCMCData(self):
        if self.saveFolder is not None:
            for i in range(self.mcmcNumWalks):
                try:
                    f = self.saveFolder + os.sep + (self.mcmcDataFile % i)
                    self._debug("Trying to load walk number %d from %s" % (i, f))
                    walk = np.loadtxt(f)
                    while len(self.mcmcData) < i:
                        self._debug("Having to fill in an empty walk")
                        self.mcmcData.append(self._getNewWalk())
                        self._saveMCMCData(len(self.mcmcData) - 1)
                    self.mcmcData.append(walk)
                    n = self.mcmcData[i][:,0].size
                    if (n < self.mcmcPoints):
                        zeros = np.zeros((self.mcmcPoints - n, self.cambNumParams + 2))
                        zeros[:,-2] -= 1
                        self.mcmcData[i] = np.concatenate((self.mcmcData[i], zeros))
                        self._saveMCMCData(i)
                except:
                    self._debug("No file (or bad file) for walk %d" % i)
                    pass
        
    def _debug(self, string):
        if self.debug:
            print string
            sys.stdout.flush()
    
    def _setupDataStructures(self):
        self._debug('Setting up data structures')
        sys.stdout.flush()
        if self.gridActive: self._loadGridData()
        if self.mcmcActive: self._loadMCMCData()
        self._debug('Data structures set up')
        
    def clearFiles(self):
        if (os.path.exists(self.saveFolder)):
            for file in os.listdir(self.saveFolder):
                if file.endswith(".txt"):
                    os.remove(self.saveFolder + os.sep + file)
                    print "Removed file: %s" % file

    def _callCamb(self, variables):
        forCamb = copy.deepcopy(self.cambArgs)
        for i, v in enumerate(variables):
            forCamb[self.cambParams[i]['name']] = v
        try:
            return self.camb(**forCamb)
        except KeyboardInterrupt:
            raise
        except:
            proc = psutil.Process()
            print proc.open_files()
            time.sleep(360)
            return self._callCamb(variables)
            
    
    def _getChi2(self, variables):
        self._debug("Variables: %s" % variables)
        cambResult = self._callCamb(variables)
        tempR = cambResult[self.dataSource]
        if (self.dataAutoCorrelation):
            model = methods.pk2xi(tempR['k/h'], tempR['power'], self.dataX)
            chi2s = np.dot((self.dataY - model).T, np.dot(self.covariance, self.dataY - model))
        else:
            model = interp1d(tempR[self.columnX], tempR[self.columnY])(self.dataX)
            chi2s = ((self.dataY - model) / self.dataE)
            chi2s *= chi2s
        return np.sum(chi2s)
    
    def _computeGrid(self):
        self._debug("Computing Grid")
        c = 0
        for i in range(self.gridData[:,0].size):
            row = self.gridData[i]
            if (row[-1] == -1):
                row[-1] = self._getChi2(row[:-1])
                c += 1
                if (self.saveFolder and (c % self.saveFrequencyGrid == 0 or i == self.gridData[:,0].size - 1)):
                    self._saveGridData()
        self._debug('Computing grid finished')
        

    def getUsable(self, walk):
        b = np.floor(self.mcmcBurnIn * walk[:,0].size)
        return walk[b::self.mcmcThinning,:-1]
        
    def getFinalData(self, grid=True, mcmc=True, trimmed=0):
        if self.finalData is not None and grid and mcmc and trimmed == 0:
            return self.finalData
        data = None
        if grid:
            data = self.gridData
        if mcmc:
            for walk in self.mcmcData:
                final = self.getUsable(walk)
                if data is None:
                    data = final
                else:
                    data = np.concatenate((data, final))
        if trimmed == 0:
            return data
        else:
            return data[data[:,-1].argsort()][:trimmed,:]
        
    # DATA GRID IS HERE
    # DATA GRID IS HERE
    # DATA GRID IS HERE
    def getDataGrid(self, grid=True, mcmc=True):
        if self.finalDataGrid is not None and grid and mcmc:
            return self.finalDataGrid
        mesh = np.meshgrid(*[i['grid2'] for i in self.cambParams], indexing='ij')
        data = self.getFinalData(grid, mcmc)
        if self.cambNumParams > 2:
            data = data[data[:,-1].argsort()]
            thing = KDTreeInterpolator(data[:,:-1], data[:,-1])
            positions = np.vstack([i.ravel() for i in mesh]).T
            k = 30
            chi2 = thing(positions, k=k, eps=100).reshape(tuple([i['grid2'].size for i in self.cambParams]))
        else:
            chi2 = griddata(data[:,:-1], data[:,-1], tuple(mesh), method='cubic') 
        #chi2 = rbfi(*mesh)
        return (mesh, chi2)
                
    def getStartOfWalk(self):
        result = []
        for i in range(self.cambNumParams):
            result.append(np.random.uniform(self.cambParams[i]['min'], self.cambParams[i]['max']))
        chi2 = self._getChi2(result)
        result.append(chi2)
        result.append(1)
        return result
                
    def accept(self, newChi2, oldChi2):
        prob = np.exp((oldChi2*oldChi2 - newChi2*newChi2)/2)
        u = np.random.uniform()
        if prob > u:
            self._debug("ACCEPT! %0.8f > %0.8f... chi2s %0.5f vs %0.5f" % (prob, u, newChi2, oldChi2))
        else:
            self._debug("REJECT! %0.8f < %0.8f... chi2s %0.5f vs %0.5f" % (prob, u, newChi2, oldChi2))
        return prob > u
                
    def walkFrom(self, walkPoint):
        accept = 0
        while True:
            coords = []
            for i in range(self.cambNumParams):
                coords.append(np.random.normal(walkPoint[i], self.cambParams[i]['sigma']))
            accept += 1
            chi2 = self._getChi2(coords)
            if (self.accept(chi2, walkPoint[-2])):
                coords.append(chi2)
                coords.append(accept)
                return coords
            elif accept > 100:
                print "%d rejected steps in a row!!!" % accept
    def _doWalk(self, index):
        walk = self.mcmcData[index]
        for i in range(walk[:,0].size):
            row = walk[i]
            if row[-2] == -1:
                if i == 0:
                    walk[i] = self.getStartOfWalk()
                    self._debug("Start walk: %s" % walk[i])
                    self._saveMCMCData(index)
                else:
                    walk[i] = self.walkFrom(walk[i - 1])
                    self._debug("Walk to: %s" % walk[i])
                    if (i % self.saveFrequencyMCMC == 0 or i == walk[:,0].size - 1):
                        self._saveMCMCData(index)
                
    def _computeMCMC(self):
        self._debug("Computing MCMC")
        for i in range(len(self.mcmcData)):
            self._doWalk(i)
        while len(self.mcmcData) < self.mcmcNumWalks:
            self._debug("Starting new walk")
            self.mcmcData.append(self._getNewWalk())
            self._doWalk(len(self.mcmcData) - 1)
        self._debug('Computing MCMC finished')
    

    
    def getMinChi2(self):
        if self.finalDataGrid is None:
            self.finalDataGrid = self.getDataGrid()
        minChi2 = np.nanmin(self.finalDataGrid[-1])
        loc = np.where(self.finalDataGrid[-1] == minChi2)
        params = [v['grid2'][loc[i][0]] for i,v in enumerate(self.cambParams)]
        return (minChi2, params)
        

            
    def plotData(self,  size=(17,9)):
        fig = plt.figure(figsize=size, dpi=300)
        matplotlib.rcParams.update({'font.size': 14})
        ax0 = fig.add_subplot(1,1,1)
        ax0.plot(self.dataX, self.dataY, color='b')
        ax0.errorbar(self.dataX, self.dataY,yerr=(self.dataE), linestyle="None", color='b')

    def indexFromName(self, name):
        r = [i for i,v in enumerate(self.cambParams) if v['name'] == name]
        if len(r) == 0:
            raise Exception("Name %s not found" % name)
        if len(r) > 1:
            raise Exception("Name %s found %d times" % (name, len(r)))
        return r[0]

    def getSummedData(self, *colNames):
        (mesh, chi2) = self.getDataGrid()
        index = [self.indexFromName(i) for i in colNames]
        index.sort()
        sumAxes = [i for i in range(self.cambNumParams) if i not in index]
        if len(sumAxes) == 0:
            return (mesh, chi2, [self.indexFromName(i) for i in colNames])
        else:
            sys.stdout.flush()
            mesh = np.meshgrid(*[self.cambParams[i]['grid2'] for i in index], indexing='ij')
            summed = np.sum(chi2, axis=tuple(sumAxes))
            summed /= np.nanmin(summed)
            return (mesh, summed, index)
            
            
    def showContour(self, x, y, grid=True, mcmc=True, sigmas=None, final=False, size=(13,9)):
        print "Warning, this only works for 2D parameter spaces at the moment, no marginalisation"
        if sigmas is None:
            sigmas = np.array([0.2, 0.5, 1,2,3,4,5,10,20,30,300,3000])
            
        fig = plt.figure(figsize=size, dpi=300)
        matplotlib.rcParams.update({'font.size': 14})
        ax0 = fig.add_subplot(1,1,1)
        levels = methods.getDeltaChi2(self.cambNumParams, sigmas)
        index = [self.indexFromName(x), self.indexFromName(y)]
        index.sort()
        if self.cambNumParams > 2 or final:
            (mesh, chi2, index) = self.getSummedData(x, y)
            chi2 = chi2 - np.nanmin(chi2)
            ax0.set_xlabel(self.cambParams[index[0]]['name'])
            ax0.set_ylabel(self.cambParams[index[1]]['name'])
        else:
            (mesh, chi2) = self.getDataGrid(grid, mcmc)
            chi2 = chi2 - np.nanmin(chi2)
            ax0.set_xlabel(self.cambParams[index[0]]['name'])
            ax0.set_ylabel(self.cambParams[index[1]]['name'])
        cs = ax0.contour(mesh[0], mesh[1], chi2, levels=levels, cmap=plt.cm.gnuplot2)
        #cs = ax0.contour(mesh[0], mesh[1], chi2, cmap=plt.cm.gnuplot2)
        fmt = {}
        for l, s in zip(cs.levels, sigmas):
            fmt[l] = '$%0.1f\sigma$' % s
        ax0.clabel(cs, inline=1, fontsize=12, fmt=fmt)
        ax0.set_xlim([self.cambParams[index[0]]['grid2'][0], self.cambParams[index[0]]['grid2'][-1]])
        ax0.set_ylim([self.cambParams[index[1]]['grid2'][0], self.cambParams[index[1]]['grid2'][-1]])
        #ax0.clabel(cs, inline=1, fontsize=12)

        return ax0
        
    def plotPaths(self, x, y, grid=True, mcmc=True, filtered=False, final=False):
        ax0 = self.showContour(x, y, grid=grid, mcmc=mcmc, final=final)
        index = [self.indexFromName(x), self.indexFromName(y)]
        index.sort()
        for walk in self.mcmcData:
            if filtered: 
                walk = self.getUsable(walk)
            ax0.plot(walk[:, index[0]], walk[:,index[1]], 'o-', alpha=0.05)
        ax0.plot(self.gridData[:, index[0]], self.gridData[:, index[1]], 'x', color='c')
        
        
    def plot3DScatter(self, x, y, z, size=(13, 9), spin=50, elevation=20, grid=True, final=False, mcmc=True):
        if self.cambNumParams < 3:
            print "Imma gonna ask you to use the 2d plot"
            return
        if self.cambNumParams > 3 and final == False:
            print "When showing 3D data for tests with more than 3 variables, you need to set final=True"
            print "as it is not possible to marginalise over the grid and mcmc results"
            return
        fig = plt.figure(figsize=size, dpi=300)
        matplotlib.rcParams.update({'font.size': 14})
        ax0 = fig.add_subplot(1,1,1, projection='3d')
        if self.cambNumParams > 3 or final:
            (mesh, chi2, index) = self.getSummedData(x, y, z) 
            print "GRR"
            sys.stdout.flush()
            dx = mesh[0]
            dy = mesh[1]
            dz = mesh[2]
            dc = np.log(chi2)
            ax0.set_xlabel(self.cambParams[index[0]]['name'])
            ax0.set_ylabel(self.cambParams[index[1]]['name'])
            ax0.set_zlabel(self.cambParams[index[2]]['name'])
        else:
            data = self.getFinalData(grid=grid, mcmc=mcmc)
            dx = data[:,self.indexFromName(x)]
            dy = data[:,self.indexFromName(y)]
            dz = data[:,self.indexFromName(z)]
            ax0.set_xlabel(x)
            ax0.set_ylabel(y)
            ax0.set_zlabel(z)
            dc = np.log(data[:,-1])
        dcr = [np.nanmin(dc), np.nanmax(dc)]
        alphas = (1 - (dc - dcr[0])/(dcr[1]-dcr[0]))**4 * 200   
        sys.stdout.flush()
        s = ax0.scatter(dx, dy, dz, c=dc, s=alphas, alpha=0.2)
        sys.stdout.flush()
        s.set_clim(dcr)
        cb = fig.colorbar(s)
        cb.set_label('$\log(\chi^2)$', fontsize=20)
        ax0.view_init(elev=elevation, azim=spin)

    def getFinalOutputModel(self):
        if self.finalOutputModel is not None:
            return self.finalOutputModel
            
        minChi2, params = self.getMinChi2()
        model = self._callCamb(params)[self.dataSource]
        x = model[self.columnX]
        y = model[self.columnY]
        if (self.dataAutoCorrelation):
            print "doom"
            ss = np.arange(1, 250, 0.2)
            y = methods.pk2xi(x, y, ss)
            x = ss
        return (x, y)
        
    def plotFinalModel(self, size=(13, 9), secondary=None):
        fig = plt.figure(figsize=size, dpi=300)
        matplotlib.rcParams.update({'font.size': 14})
        ax0 = fig.add_subplot(1,1,1)
        
        if secondary is not None:
            ax0.plot(secondary[0], secondary[0] * secondary[0] * secondary[1], color='k', label=secondary[2])
        
        minChi2, params = self.getMinChi2()
        (x, y) = self.getFinalOutputModel()
        label = 'Best fit'
        for i, v in enumerate(self.cambParams):
            label += ", %s=%0.4f" % (v['name'], params[i])
        ax0.plot(x, x * x * y, color='r', linestyle='--', linewidth=2.0, label=label)
        ax0.scatter(self.dataX, self.dataX * self.dataX * self.dataY, color='g', label="Input data points")
        ax0.errorbar(self.dataX, self.dataX * self.dataX * self.dataY,yerr=(self.dataX * self.dataX * self.dataE), linestyle="None", color='g')
        
        minX = np.min(self.dataX)
        maxX = np.max(self.dataX)
        
        xLim1 = minX - (maxX - minX) * 0.1
        xLim2 = maxX + (maxX - minX) * 0.1
        ax0.set_xlim([xLim1, xLim2])
        ax0.set_ylabel(self.columnY)
        ax0.set_xlabel(self.columnX)
        ax0.legend(loc='lower right')
        
    def getHistogramData(self, param, bins=50, size=(13, 9)):
        sys.stdout.flush()
        data = self.getFinalData(grid=False)
        i = self.indexFromName(param)
        hist, bins = np.histogram(data[:,i], bins=bins)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        fig = plt.figure(figsize=size, dpi=300)
        matplotlib.rcParams.update({'font.size': 14})
        ax0 = fig.add_subplot(1,1,1)
        ax0.bar(center, hist, align='center', width=width)
        ax0.set_xlabel(param)
        ax0.set_ylabel("Counts")
        #ax0.xlim(min(binEdges), max(binEdges))
        #return (H, edges)
        
    def start(self):
        if self.dataX is None:
            print "You need to load in data to test against, call setPhysicalData"
            return
        if len(self.cambParams) == 0:
            print "There is nothing to iterate over, call addVariable"
            return
        self._setupDataStructures()
        if self.gridActive: self._computeGrid()
        if self.mcmcActive: self._computeMCMC()
        self.finalData = self.getFinalData()
        self.finalDataGrid = self.getDataGrid()
        #self.finalOutput = self.getMinChi2()
        self.finalOutputModel = self.getFinalOutputModel()
        
        