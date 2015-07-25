from __future__ import print_function
from sam import *
import numpy as np
import sys
from multiprocessing import Pool
import matplotlib, matplotlib.pyplot as plt
import time
import matplotlib.gridspec as gridspec
import matplotlib.patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats
from matplotlib.path import Path
from matplotlib import rc
from pylab import *
import pickle
import matplotlib.lines as mlines
from scipy.interpolate import *


class CambMCMCManager(object):
    def __init__(self, uid, fitter, debug=False):
        self.uid = uid
        self.fitter = fitter
        self._debug = debug
        self.configureMCMC()
        self.configureSaving()
        self.steps = {}
        self.slowGenerator = None
        self.finalBounds = {}
        self.allSteps = {}
        self.finalSteps = {}
    """ Stuff that people all call """
    def configureMCMC(self, numStarts=200, sigmaCalibrationLength=50, calibrationLength=1000, numCalibrations=9, desiredAcceptRatio=0.3, thinning=2, maxSteps=100000):
        self.sigmaCalibrationLength = sigmaCalibrationLength
        self.calibrationLength = calibrationLength
        self.numCalibrations = numCalibrations
        self.desiredAcceptRatio = desiredAcceptRatio
        self.thinning = thinning
        self.maxSteps = maxSteps
        self.numStarts = numStarts
        
    def configureSaving(self, stepsPerSave=100):
        self.stepsPerSave = stepsPerSave
        
    def debug(self, string):
        if self._debug:
            print(string)
    
    
    def doWalk(self, index=0, outputToFile=False, chunkLength=None):           
        cambMCMC = CambMCMC(self)
        result = cambMCMC(index, outputToFile, chunkLength=chunkLength)
        return result
        
    def consolidateData(self, uid=None):
        if uid is None: uid = self.uid
        print(uid)
        self.steps[uid] = {}
        self.allSteps[uid] = None
        self.finalSteps[uid] = None
        index = 0
        while (True):
            steps = self.loadSteps(index, uid=uid, fillZeros=False)
            if steps is None:
                break
            else:
                self.steps[uid][index] = steps
                if self.allSteps[uid] is None:
                    self.allSteps[uid] = np.copy(steps)
                else:
                    self.allSteps[uid] = np.vstack((self.allSteps[uid], steps))
            index += 1
        toUse = []
        for x,v in self.steps[uid].iteritems():
            if v[:,0].size > self.calibrationLength * self.numCalibrations:
                dataArray = v[self.calibrationLength * self.numCalibrations :: self.thinning, :]
                dataArray[:,2] = x
                toUse.append(dataArray)
        if len(toUse) > 0:
            self.finalSteps[uid] = np.concatenate(tuple(toUse))
            
        if self.finalSteps[uid] is not None:
            self.debug("Consolidated data contains %d data points" % self.finalSteps[uid].shape[0])
        else:
            self.debug("Consolidated data is empty")
    
    def testConvergence(self, uid=None):
        if uid is None: uid = self.uid
        finals = [self.finalSteps[uid][self.finalSteps[uid][:,2] == i, 3:] for i in np.unique(self.finalSteps[uid][:,2])]
        variances = [np.var(f, axis=0, ddof=1) for f in finals]
        w = np.mean(variances, axis=0)
    
        allMean = np.mean(self.finalSteps[uid][:,3:], axis=0)
        walkMeans = [np.mean(f, axis=0) for f in finals]
        b = np.zeros(len(allMean))
        
        for i, walkMean in enumerate(walkMeans):
            b += 1.0 * (1.0 / (len(walkMeans) - 1)) * finals[i][:,0].size * np.power((walkMean - allMean), 2)
    
        meanN = 0
        for f in finals:
            meanN += f[:,0].size
        meanN *= 1.0 / len(finals)
        
        estVar = (1 - (1.0 / meanN)) * w + (1.0 / meanN) * b
        r = np.sqrt(estVar / w)
        print("estvar and r coming up")
        print(estVar)
        print(r)
        return estVar, r
        
    def getParameterBounds(self, numBins=50, uid=None):
        if uid is None: uid = self.uid
        interpVals = np.array([0.15865, 0.5, 0.84135])
        res = []
        string = "\n\n area \n\n\\begin{align}\n"
        if self.finalSteps.get(uid) is None:
            return None
        for paramList in self.fitter.getParams():
            param = paramList[0]
            index = self.fitter.getIndex(param)
            
            allData = self.allSteps[uid][:, index]
            data = self.finalSteps[uid][:, index]
            hist, bins = np.histogram(data, bins=numBins)
            centers = (bins[:-1] + bins[1:]) / 2
            dist = 1.0 * hist.cumsum() / hist.sum()
            bounds = interp1d(dist, centers, bounds_error=False)(interpVals)
            maxL = allData[self.allSteps[uid][:,0].argmin()]
            maxLchi2 = self.allSteps[uid][:,0].min()
            res.append((param, maxL, bounds))
            string += "%s &= %0.3f^{+%0.3f}_{-%0.3f} \\\\ \n" % (paramList[3].replace("$", ""), bounds[1], bounds[2]-bounds[1], bounds[1]-bounds[0])
            print("%s, maxL=%0.4f with chi2=%0.3f, 1sigmaDistBounds=(%0.4f, %0.4f, %0.4f)" % (param, maxL, maxLchi2, bounds[0], bounds[1], bounds[2]))
        string += "\\end{align}\n"
        print(string)
        self.finalBounds[uid] = res
        return res
    
    def getTamParameterBounds(self, numBins=50, desiredArea=0.6827, uid=None, kind='cubic'):
        if uid is None: uid = self.uid
        res = []
        string = "\n\ntam\n\n\\begin{align}\n"
        if self.finalSteps.get(uid) is None:
            return None
        for paramList in self.fitter.getParams():
            param = paramList[0]
            index = self.fitter.getIndex(param)
            
            allData = self.allSteps[uid][:, index]
            data = self.finalSteps[uid][:, index]
            hist, bins = np.histogram(data, bins=numBins, density=True)
            centers = (bins[:-1] + bins[1:]) / 2
            
            xs = np.linspace(centers[0], centers[-1], 10000)
            ys = interp1d(centers, hist, kind=kind)(xs)
            plt.figure()
            plt.plot(xs,ys)
            yscumsum = ys.cumsum() #interp1d(centers, dist)(xs)
            yscumsum /= yscumsum.max()
            indexes = np.arange(xs.size)
            
            startIndex = ys.argmax()
            maxVal = ys[startIndex]
            minVal = 0
            threshold = 0.001
            
            x1 = None
            x2 = None
            count = 0
            while x1 is None:
                mid = (maxVal + minVal) / 2.0
                count += 1                    
                try:
                    if count > 100:
                        print("We goofed up")
                        raise Exception("no")
                    i1 = indexes[ys[:startIndex] > mid][0]
                    i2 = startIndex + indexes[ys[startIndex:] < mid][0]
                except:
                    x1 = 0
                    x2 = 0
                    print("Parameter %s is not constrained" % param)
                area = yscumsum[i2] - yscumsum[i1]
                a = np.abs(area - desiredArea)
                #print(maxVal, minVal, area)
                if a < threshold:
                    x1 = xs[i1]
                    x2 = xs[i2]
                elif area < desiredArea:
                    maxVal = mid
                elif area > desiredArea:
                    minVal = mid

            bounds = np.array([x1, xs[startIndex], x2])
            maxL = allData[self.allSteps[uid][:,0].argmin()]
            maxLchi2 = self.allSteps[uid][:,0].min()
            res.append((param, maxL, bounds))
            string += "%s &= %0.3f^{+%0.3f}_{-%0.3f} \\\\ \n" % (paramList[3].replace("$", ""), bounds[1], bounds[2]-bounds[1], bounds[1]-bounds[0])
            print("%s, maxL=%0.4f with chi2=%0.3f, 1sigmaDistBounds=(%0.4f, %0.4f, %0.4f)" % (param, maxL, maxLchi2, bounds[0], bounds[1], bounds[2]))
        string += "\\end{align}\n"
        print(string)
        self.finalBounds[uid] = res
        return res
            
    def getBlakeParameterBounds(self, numBins=50, uid=None):
        if uid is None: uid = self.uid
        interpVals = np.array([0.15865, 0.84135])
        res = []
        string = "\n\n blake \n\n \\begin{align}\n"
        if self.finalSteps.get(uid) is None:
            return None
        for paramList in self.fitter.getParams():
            param = paramList[0]
            index = self.fitter.getIndex(param)
            
            allData = self.allSteps[uid][:, index]
            data = self.finalSteps[uid][:, index]
            hist, bins = np.histogram(data, bins=numBins)
            centers = (bins[:-1] + bins[1:]) / 2
            dist = 1.0 * hist.cumsum() / hist.sum()
            bounds = interp1d(dist, centers, bounds_error=False)(interpVals)
            bounds = np.array([bounds[0], np.mean(bounds), bounds[1]])
            maxL = allData[self.allSteps[uid][:,0].argmin()]
            maxLchi2 = self.allSteps[uid][:,0].min()
            res.append((param, maxL, bounds))
            string += "%s &= %0.3f^{+%0.3f}_{-%0.3f} \\\\ \n" % (paramList[3].replace("$", ""), bounds[1], bounds[2]-bounds[1], bounds[1]-bounds[0])
            print("%s, maxL=%0.4f with chi2=%0.3f, 1sigmaDistBounds=(%0.4f, %0.4f, %0.4f)" % (param, maxL, maxLchi2, bounds[0], bounds[1], bounds[2]))
        string += "\\end{align}\n"
        print(string)
        self.finalBounds[uid] = res
        return res

    def plotWalk(self, param1, param2, final=True, size=(13,9), uid=None):
        if uid is None: uid = self.uid
        fig = plt.figure(figsize=size, dpi=300)
        matplotlib.rcParams.update({'font.size': 14})
        ax0 = fig.add_subplot(1,1,1)
        
        if final:
            for i in np.unique(self.finalSteps[uid][:, 2]):
                d = self.finalSteps[uid][self.finalSteps[uid][:,2] == i]
                ax0.plot(d[:, self.fitter.getIndex(param1)], d[:, self.fitter.getIndex(param2)], 'o', markersize=5, alpha=0.05)
        else:
            for x in self.steps[uid]:
                ax0.plot(self.steps[uid][x][self.steps[uid][x][:,0] >= 0, self.fitter.getIndex(param1)], self.steps[uid][x][self.steps[uid][x][:,0] >= 0, self.fitter.getIndex(param2)], 'o-', markersize=5, alpha=0.05)
        ax0.set_xlabel(param1)
        ax0.set_ylabel(param2)
        
    def getHistogram(self, param, bins=50, size=(13,9), uid=None):
        ''' Because doSteps won't finish, you run it and then ask for results when you are happy with the dataset'''
        if uid is None: uid = self.uid

        data = self.finalSteps[uid][:, self.fitter.getIndex(param)]
        hist, bins = np.histogram(data, weights=self.finalSteps[uid][:, 1], bins=bins)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        
        fig = plt.figure(figsize=size, dpi=300)
        matplotlib.rcParams.update({'font.size': 14})
        ax0 = fig.add_subplot(1,1,1)
        ax0.bar(center, hist, align='center', width=width)
        ax0.set_xlabel(param)
        ax0.set_ylabel("Counts")
        
    def plotResults(self, size=(15,15), filename=None, uid=None, plotLine=True):
        if uid is None: uid = self.uid
        fig = plt.figure(figsize=size, dpi=300)
        matplotlib.rcParams.update({'font.size': 14})
        matplotlib.rcParams['axes.labelsize'] = 20
        rc('text', usetex=False)
        matplotlib.rcParams['xtick.labelsize'] = 14
        matplotlib.rcParams['ytick.labelsize'] = 14
        params = []
        for p in self.fitter.getParams():
            #if p[0] != "sigmav":
            params.append(p)
        
        n = len(params)
        gs = gridspec.GridSpec(n, n)
        gs.update(wspace=0.0, hspace=0.0) 
        
        for i in range(n):
            for j in range(i + 1):
                ax = fig.add_subplot(gs[i,j])
                pi = params[i]
                pj = params[j]
                if i == j:
                    self.plotBars(ax, pi[0], False, i==n-1, uid=uid)
                else:
                    self.plotContour(ax, pi[0], pj[0], j==0, i==n-1, uid=uid, plotLine=plotLine)
                if i == n - 1:
                    ax.set_xlabel(pj[3])
                if j == 0:
                    ax.set_ylabel(pi[3])
                    
        if filename is not None:
            fig.savefig("%s.png" % filename, bbox_inches='tight', dpi=100, transparent=True)
            fig.savefig("%s.pdf" % filename, bbox_inches='tight', dpi=600, transparent=True)
    

    def plotOmch2AndAlpha(self, size=(7,7), filename=None, uids=None, linewidth=2, degeneracies=True, labels=None, styles=None, alpha=None, omch2=None):
        if uids is None: uids = [self.uid]
        
        fig = plt.figure(figsize=size, dpi=300)
        matplotlib.rcParams.update({'font.size': 14})
        matplotlib.rcParams['axes.labelsize'] = 20
        rc('text', usetex=False)
        matplotlib.rcParams['xtick.labelsize'] = 14
        matplotlib.rcParams['ytick.labelsize'] = 14
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace=0.0, hspace=0.0) 
        ax0 = fig.add_subplot(gs[1:5, 0:4])
        ax1 = fig.add_subplot(gs[0, 0:4])
        ax2 = fig.add_subplot(gs[1:5, 4])
        
        if styles is None:
            styles = ['-'] * len(uids)
        
        if degeneracies:
            omch2s = np.linspace(0.05, 0.3, 100)
            alphas = np.linspace(0.7, 1.3, 100)
            k = 0.34
            constAcoustic = k / np.sqrt(omch2s)
            ax0.plot(omch2s, constAcoustic, '--', color='k', label="Constant A")
            
        if alpha is not None:
            ax0.plot([-100, 100], [alpha, alpha], '--', color='k', alpha=0.7)
        if omch2 is not None:
            ax0.plot([omch2, omch2], [-100, 100], '--', color='k', alpha=0.7)
        
        colors = ["#1E88E5", "#D32F2F", "#4CAF50", "#673AB7", "#FFC107", "#795548", "#64B5F6", "#8BC34A", "#757575", "#CDDC39"]
        if labels is None:
            labels = uids
        handles = []
        for i, uid in enumerate(uids):
            handles.append(self.plotContour(ax0, 'alpha', 'omch2', True, True, style=styles[i], plotLine=False, ticks=6, uid=uid, contourf=False, color=colors[i],linewidth=linewidth))
            self.plotBars(ax1, 'omch2', False, False, bar=False, uid=uid, style=styles[i], color=colors[i],linewidth=linewidth)
            self.plotBars(ax2, 'alpha', False, False, sideways=True, bar=False, uid=uid, style=styles[i], color=colors[i],linewidth=linewidth)
            
        ax0.legend(handles=[mlines.Line2D([], [], linestyle=s, color=c, label=g) for (c,g, s) in zip(colors, labels, styles)])
        
        ax0.set_xlabel(r"$\Omega_c h^2$")
        ax0.set_ylabel(r"$\alpha$")
        
        if filename is not None:
            #fig.savefig("%s.png" % filename, bbox_inches='tight', dpi=300, transparent=True)
            fig.savefig("%s.pdf" % filename, bbox_inches='tight', transparent=True)
        
    def plotBars(self, ax, param, showylabel, showxlabel, bins=50, sideways=False, bar=True, style='-', ticks=4, uid=None, color="#47a0e7", linewidth=1):
        if uid is None: uid = self.uid
        self.debug("Plotting bars for %s" % param)
        data = self.finalSteps[uid][:, self.fitter.getIndex(param)]
        hist, bins = np.histogram(data, weights=self.finalSteps[uid][:, 1], bins=bins, density=True)
        width = 1 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        if sideways:
            if bar:
                ax.barh(center, hist, align='center', height=width, edgecolor="none", facecolor="#333333")
            else:
                ax.plot(hist, center, style, color=color, linewidth=linewidth)                
            ax.set_ylim([min(data), max(data)])
        else:
            if bar:
                ax.bar(center, hist, align='center', width=width, edgecolor="none", facecolor="#333333")
            else:
                ax.plot(center, hist, style, color=color, linewidth=linewidth)
            ax.set_xlim([min(data), max(data)])
        ax.yaxis.set_major_locator(plt.MaxNLocator(ticks))
        ax.xaxis.set_major_locator(plt.MaxNLocator(ticks))
        if not showylabel:
            ax.set_yticklabels([])
        if not showxlabel:
            ax.set_xticklabels([])
        return (center, hist)
    
        
    def plotContour(self, ax, param1, param2, showylabel, showxlabel, levels=[0, 0.6827, 0.9545], linewidth=1, style='-', color="#666666", plotLine=True, contourf=True, ticks=4, uid=None):
        if uid is None: uid = self.uid
        if style == '-':
            style = 'solid'
        elif style == '--':
            style = 'dashed'
        elif style == ':':
            style = 'dotted'
        self.debug("Plotting contour for %s and %s" % (param1, param2))
        
        ys = self.finalSteps[uid][:, self.fitter.getIndex(param1)]
        xs = self.finalSteps[uid][:, self.fitter.getIndex(param2)]
        minIndex = self.finalSteps[uid][:,0].argmin()
        bins = np.sqrt(len(xs) / 30);
        bins = 50
        L_MCMC, xBins, yBins = np.histogram2d(xs, ys, weights=self.finalSteps[uid][:, 1], bins=bins)
        L_MCMC[L_MCMC == 0] = 1E-16  # prevents zero-division errors
        vals = self.convert_to_stdev(L_MCMC.T)
        
        if contourf: cf = ax.contourf(0.5 * (xBins[:-1] + xBins[1:]), 0.5 * (yBins[:-1] + yBins[1:]), vals, levels=levels, colors=("#BBDEFB", "#90CAF9"))
        c = ax.contour(0.5 * (xBins[:-1] + xBins[1:]), 0.5 * (yBins[:-1] + yBins[1:]), vals, levels=levels, linestyles=style, linewidths=linewidth, colors=color)
        ax.scatter(xs[minIndex], ys[minIndex], color=color, s=20)
        if plotLine:
            if param2 == 'omch2':
                ax.plot([0.113, 0.113], [min(ys), max(ys)], '--', color="#444444")
            elif param1 == 'omch2':
                ax.plot([min(xs), max(xs)], [0.113, 0.113], '--', color='#444444')
        ax.set_xlim([min(xs), max(xs)])
        ax.set_ylim([min(ys), max(ys)])
        ax.yaxis.set_major_locator(plt.MaxNLocator(ticks))
        ax.xaxis.set_major_locator(plt.MaxNLocator(ticks))
        if not showylabel:
            ax.set_yticklabels([])            
        if not showxlabel:
            ax.set_xticklabels([])
        
        if contourf:
            return cf
        else:
            return c
            
    def convert_to_stdev(self, sigma):
        """
        From astroML
        
        Given a grid of log-likelihood values, convert them to cumulative
        standard deviation.  This is useful for drawing contours from a
        grid of likelihoods.
        """
        #sigma = np.exp(logL)
    
        shape = sigma.shape
        sigma = sigma.ravel()
    
        # obtain the indices to sort and unsort the flattened array
        i_sort = np.argsort(sigma)[::-1]
        i_unsort = np.argsort(i_sort)
    
        sigma_cumsum = 1.0* sigma[i_sort].cumsum()
        sigma_cumsum /= sigma_cumsum[-1]
    
        return sigma_cumsum[i_unsort].reshape(shape)
        

    def getSaveFileName(self, index, uid=None, cov=False):
        if uid is None: uid = self.uid
        return "cambMCMC_%s_%s_%d.npy" % (uid, 'cov' if cov else 'steps', index)
        
    def loadSteps(self, index, uid=None, fillZeros=True):
        if uid is None: uid = self.uid
        try:
            res = np.load(self.getSaveFileName(index, uid=uid))
            self.debug("Save file found containing %d steps" % res[:,0].size)
            if (fillZeros and res[:,0].size < self.maxSteps):
                zeros = np.zeros((self.maxSteps - res[:,0].size, 3 + self.fitter.getNumParams()))
                zeros[:, 0] -= 1
                return np.concatenate((res, zeros))
            else:
                return res
        except:
            self.debug("No save file found")
            if fillZeros:
                zeros = np.zeros((self.maxSteps, 3 + self.fitter.getNumParams()))
                zeros[:, 0] -= 1
                return zeros
            else:
                return None


    def plotMostLikelyModel(self, size=(11,7), filename=None, uid=None):
        if uid is None: uid = self.uid
        if self.finalBounds[uid] is None:
            self.getParameterBounds(uid=uid)
        ss = np.arange(10,200,1)
        params = [x[1] for x in self.finalBounds[uid]]
        meanParams = [x[2][1] for x in self.finalBounds[uid]]
        
        datax, datay, datae, dataqy, dataqe, z = self.fitter.getData()
        rawx, rawy, rawe, rawqy, rawqe, z = self.fitter.getRawData()
                
        
        model = self.fitter.getModel(params, ss)
        meanModel = self.fitter.getModel(meanParams, ss)
        if self.fitter.matchQuad:
            mp = model[:model.size/2]
            qp = model[model.size/2 :]
            meanmp = meanModel[:meanModel.size/2]
            meanqp = meanModel[meanModel.size/2 :]
            dataqy = dataqy.T
            rawqy = rawqy.T
        else:
            mp = model
            meanmp = meanModel
      
        datay = datay.T
        rawy = rawy.T

        
        fig = plt.figure(figsize=size, dpi=300)
        ax0 = fig.add_subplot(1,1,1)

        matplotlib.rcParams.update({'font.size': 14})
        matplotlib.rcParams['axes.labelsize'] = 20
        rc('text', usetex=False)
        matplotlib.rcParams['xtick.labelsize'] = 16
        matplotlib.rcParams['ytick.labelsize'] = 16

        ax0.plot(ss, ss*ss*mp, label="Most Likely Model", color='b')
        ax0.plot(ss, ss*ss*meanmp, label="Mean Model", color='r')
        if self.fitter.matchQuad:
            ax0.plot(ss, ss*ss*qp, '--', color='b')
            ax0.plot(ss, ss*ss*meanqp, '--', color='r')
        ax0.errorbar(rawx, rawx*rawx*rawy, yerr=rawx*rawx*rawe, fmt='o', label="all data", alpha=0.2, color='k')
        ax0.errorbar(datax, datax*datax*datay, yerr=datax*datax*datae, fmt='o', label="matched data", color='g')
        
        if self.fitter.matchQuad:
            ax0.errorbar(rawx, rawx*rawx*rawqy, yerr=rawx*rawx*rawqe, fmt='D', alpha=0.2, color='k')
            ax0.errorbar(datax, datax*datax*dataqy, yerr=datax*datax*dataqe, fmt='D', color='g')
        
        ax0.set_ylabel("$s^2 p(s)$", fontsize=18)
        ax0.set_xlabel("$s (h^{-1} \mathrm{Mpc})$", fontsize=18)
        plt.legend()
        
        if filename is not None:
            fig.savefig("%s.png" % filename, bbox_inches='tight', dpi=100, transparent=True)
            fig.savefig("%s.pdf" % filename, bbox_inches='tight', dpi=600, transparent=True)
            
            
            
            
            
            
            
            
            
            
            
            
class CambMCMC(object):
    
    def __init__(self, parent):
        self._debug = parent._debug
        self.parent = parent
        self.uid = parent.uid
        self.outputFile = None
        self.fitter = parent.fitter        
        self.sigmaCalibrationLength = parent.sigmaCalibrationLength
        self.calibrationLength = parent.calibrationLength
        self.numCalibrations = parent.numCalibrations
        self.desiredAcceptRatio = parent.desiredAcceptRatio
        self.thinning = parent.thinning
        self.maxSteps = parent.maxSteps
        self.stepsPerSave = parent.stepsPerSave
        self.numStarts = parent.numStarts
        
    """ Stuff the class uses to do everything """
    def debug(self, text):
        if self._debug:
            if self.outputFile is not None:
                print(text, file = self.outputFile)
            else:
                print(text)
                sys.stdout.flush()
    

    def __call__(self, index=0, fileLogging=False, chunkLength=None):

        if fileLogging:
            self.outputFile = open('cambMCMC_%s_%d_log.txt' % (self.uid, index), 'w+', 8000)
        self.debug("doMCMC for walk %d" % index)
        steps = self.parent.loadSteps(index)
        sigma, sigmaRatio, rotation = self.initialiseSigma(steps)       
        startStep = None
        die = False
        for i, step in enumerate(steps):
            #if (i != 0 and i % self.sigmaCalibrationLength == 0):
            #    tracker = SummaryTracker()
            if step[0] >= 0:
                continue
            elif startStep is None:
                startStep = i
            if i == 0:
                self.debug("Starting at beginning")
                oldParams = None
                oldChi2 = 9e9
                for x in range(self.numStarts):
                    p = self.getInitialParams()
                    c = self.fitter.getChi2(p)
                    self.debug("Got chi2 %0.2f when starting at params %s" % (c, p))
                    if c < oldChi2:
                        oldChi2 = c
                        oldParams = p
                self.debug("Best starting location of chi2 %0.2f found at %s" % (oldChi2, oldParams))
            else:
                sigmaRatio = steps[i-1, 2]
                oldParams = steps[i-1, 3:]
                oldChi2 = steps[i-1, 0]
                
            isBurnin = (i <= self.numCalibrations * self.calibrationLength)
        
            if (i != 0 and isBurnin and i % self.sigmaCalibrationLength == 0):
                sigmaRatio = self.adjustSigma(steps, i, sigmaRatio)        
                
            if (i != 0 and isBurnin and i % self.calibrationLength == 0 and i <= (self.numCalibrations-1) * self.calibrationLength):
                sigma, sigmaRatio, rotation = self.adjustCovariance(steps, i / self.calibrationLength, sigmaRatio)
                
                
            steps[i, :], oldweight = self.doStep(oldParams, oldChi2, sigma, sigmaRatio, rotation, isBurnin)
            steps[i - 1, 1] = oldweight
            self.debug("Done step %d" % i)
            
            if chunkLength is not None and (i - startStep) == chunkLength - 1:
                die = True
            
            if die or ((i + 1) % self.stepsPerSave == 0 and i > 0):
                self.saveData(index, steps)
                if die:
                    break
                
        if self.outputFile is not None:
            self.outputFile.close()
            
        return steps[steps[:,0] >= 0]
        
    def doStep(self, oldParams, oldChi2, sigma, sigmaRatio, rotation, isBurnin):
        attempts = 1
        while(True):        
            params = self.getPotentialPoint(oldParams, sigmaRatio * sigma, rotation)
            newChi2 = self.fitter.getChi2(params)

            if newChi2 is None:
                self.debug("REJECT! bounds!")
            else:
                prob = np.exp((oldChi2 - newChi2)/2)
                if prob > 1 or prob > np.random.uniform():
                    self.debug("A %0.2f v %0.2f" % (newChi2, oldChi2))
                    return np.concatenate((np.array([newChi2, 0.0, sigmaRatio]), params)), attempts
                else:
                    self.debug("R %0.2f v %0.2f" % (newChi2, oldChi2))
                    attempts += 1
                    if isBurnin and attempts >= 20 and attempts % 10 == 0:
                        sigmaRatio *= 0.9
                
        
    def saveData(self, index, steps):
        self.debug("\nSaving data")
        try:
            np.save(self.parent.getSaveFileName(index), steps[steps[:,0] >= 0]) #, fmt='%1.5e'
            self.debug("Saved data\n\n\n")
        except Exception as e:
            self.debug(e.strerror)
            
    def getPotentialPoint(self, oldParams, sigma, rotation):
        rotParams = np.dot(oldParams, rotation)
        newRotParams = rotParams + sigma * np.random.normal(size=sigma.size)
        newParams = np.dot(rotation, newRotParams)
        return newParams
        
    def initialiseSigma(self, steps):
        sigma = 0.005 * np.array([(x[2] - x[1]) for x in self.fitter.getParams()])
        sigmaRatio = 1
        rotation = np.identity(self.fitter.getNumParams())
        
        self.debug("Setting sigma, sigmaRatio and rotation to defaults")
        if (steps is not None and steps[steps[:,0] >= 0, 0].size > 0):
            s = steps[steps[:,0] > 0]
            i = s[:,0].size
            sigmaRatio = s[i-1, 2]
            
            if i % self.sigmaCalibrationLength == 0 and i < (self.numCalibrations * self.calibrationLength):
                self.debug("Calibrating sigma as well")
                sigmaRatio = self.adjustSigma(s, i, sigmaRatio) 
            if i > self.calibrationLength:
                self.debug("Calculating rotation matrix from existing steps")
                sigma, sigmaRatio, rotation = self.adjustCovariance(s, min(i / self.calibrationLength, self.numCalibrations - 1), sigmaRatio)
                if i % self.calibrationLength != 0 or i > (self.numCalibrations * (self.calibrationLength - 1)):
                    self.debug("Setting sigma ratio from data")
                    sigmaRatio = s[i-1, 2]
            if i % self.sigmaCalibrationLength == 0 and i % self.calibrationLength != 0 and i < (self.numCalibrations * self.calibrationLength):
                self.debug("Recalibrating sigma")
                sigmaRatio = self.adjustSigma(s, i, sigmaRatio) 
        
        return sigma, sigmaRatio, rotation
            
    def adjustSigma(self, steps, index, sigmaRatio):
        subsection = steps[index - self.sigmaCalibrationLength : index]
        desiredAvg = 1 / self.desiredAcceptRatio
        actualAvg = np.average(subsection[:, 1])
        n = steps[0,3:].size
        update = 1 + 2*(((desiredAvg/actualAvg) - 1)/(n))
        #update = min(1/0.7, max(0.7, update))
        ps = sigmaRatio
        sigmaRatio *= update
        self.debug("Adjusting sigma: want accept every %0.2f, got %0.2f. Updating ratio from %0.3f to %0.3f" % (desiredAvg, actualAvg, ps, sigmaRatio))
        return sigmaRatio
        
    def adjustCovariance(self, steps, index, sigmaRatio):
        if index == 1 or True:
            sigmaRatio = 0.5
        weightedAvgs, covariance = self.getCovariance(steps, index * self.calibrationLength)
        evals, evecs = self.diagonalise(covariance)
        sigma = np.sqrt(np.abs(evals)) * 2.3 / np.sqrt(evals.size)
        self.debug("[Covariance] sigma: setting sigma to %s" % (sigma))
        return sigma, sigmaRatio, evecs

    def getCovariance(self, steps, index):
        self.debug("Getting covariance")
        subset = steps[np.floor(index/2):index, :]
        print(index)
        weightedAvg = np.sum(subset[:, 3:] * subset[:, 1].reshape((subset[:,1].size, 1)), axis=0) / np.sum(subset[:,1])
        deviations = subset[:, 3:] - weightedAvg
        n = deviations[0,:].size
        covariance = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                covariance[i,j] = np.sum(deviations[:, i] * deviations[:, j] * subset[:, 1])
        covariance /= np.sum(subset[:,1])
        
        #self.debug("Found covariance: \n%s" % covariance)
        return weightedAvg, covariance
        
    def diagonalise(self, covariance):
        evals, evecs = np.linalg.eig(covariance)
        #self.debug("Diagonlised eigenvectors: \n%s" % evecs)
        return (evals, evecs)
        
    def getInitialParams(self):
        return [np.random.uniform(x[1], x[2]) for x in self.fitter.getParams()]
        
    
                