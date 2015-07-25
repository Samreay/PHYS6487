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
from scipy.interpolate import interp1d


class Fitted(object):
    def __init__(self, debug=False):
        self._debug = debug
    def getIndex(self, param):
        for i, x in enumerate(self.getParams()):
            if x[0] == param:
                return i + 3
        return None
    def debug(self, string):
        if self._debug:
            print(string)
            sys.stdout.flush()
    def getNumParams(self):
        raise NotImplementedError
    def getChi2(self, params):
        raise NotImplementedError
    def getParams(self):
        raise NotImplementedError


class CosmoMonopoleFitter(Fitted):
    def __init__(self, debug=False, cambFast=False, pk2xiFast=False):
        self._debug = debug
        self.cambFast = cambFast
        self.pk2xiFast = pk2xiFast
        self.cambDefaults = {}
        self.cambDefaults['get_scalar_cls'] = False
        if cambFast:
            self.cambDefaults['do_lensing'] = False
            self.cambDefaults['accurate_polarization'] = False
            self.cambDefaults['high_accuracy_default'] = False
            
        self.cambParams = []
        self.fitParams = [('b2', 0.001, 3, '$b^2$') ]
        
        self.ss = np.linspace(10, 211, 100)
        self.generator = None
        
    def addCambParameter(self, name, minValue, maxValue, label):
        self.cambParams.append((name, minValue, maxValue, label))
        
    def getNumParams(self):
        return len(self.cambParams) + len(self.fitParams)
        
    def addDefaultCambParams(self, **args):
        self.cambDefaults.update(args)
        
    def getParams(self):
        return self.cambParams + self.fitParams
    def generateMus(self, n=100):
        self.mu = np.linspace(0,1,n)
        self.mu2 = np.power(self.mu, 2)
        self.muNA = self.mu[np.newaxis]
        self.mu2NA = self.mu2[np.newaxis]
        self.p2 = 0.5 * (3 * self.mu2 - 1)
        self.p4 = (35*self.mu2*self.mu2 - 30*self.mu2 + 3) / 8.0
        
    def setData(self, datax, monopole, quadrupole, totalCovariance, monopoleCovariance, quadrupoleCovariance, z, matchQuad=True, minS=50, maxS=150, poles=True, angular=True, logkstar=None, log=True):
        selection = (datax > minS) & (datax < maxS)
        selection2 = np.concatenate((selection, selection))
        self.rawX = datax
        self.dataZ = z
        self.dataX = datax[selection]
        self.monopole = monopole[selection]
        if quadrupole is not None:
            self.quadrupole = quadrupole[selection]
        self.rawMonopole = monopole
        self.rawQuadrupole = quadrupole
        self.matchQuad = matchQuad
        self.poles = poles
        self.angular = angular
        self.logkstar = logkstar
        if logkstar is None:
            if log:
                self.fitParams.append(('kstar', -5, 0, '$\\log(k_*)$'))
            else:
                self.fitParams.append(('sigmav', 0, 10, '$\\sigma_v$'))
        if angular:
            self.fitParams.append(('beta', 0.1, 8, '$\\beta$'))
            self.fitParams.append(('lorentzian', 0.1, 10.0, '$\\sigma H(z)$'))
        if poles:
            self.generateMus()
            self.fitParams.append(('alpha', 0.7, 1.3, r'$\alpha_0$'))
            if matchQuad:
                self.fitParams.append(('qalpha', 0.7, 1.3, r'$\alpha_2$'))
        else:
            self.generateMus(n=50)
            self.fitParams.append(('alphaPerp', 0.7, 1.3, r'$\alpha_\perp$'))
            self.fitParams.append(('alphaParallel', 0.7, 1.3, r'$\alpha_\parallel$'))
        
        if poles and not matchQuad:
            self.totalData = self.monopole
            self.dataCov = (monopoleCovariance[:,:,3])[:,selection][selection,:]
        else:
            self.totalData = np.concatenate((self.monopole, self.quadrupole))
            self.dataCov = (totalCovariance[:,:,3])[:,selection2][selection2,:]
            
        self.rawE = np.sqrt(monopoleCovariance[np.arange(monopoleCovariance.shape[0]), np.arange(monopoleCovariance.shape[1]), 2])
        self.dataE = self.rawE[selection]
        
        if quadrupole is not None:
            self.rawQE = np.sqrt(quadrupoleCovariance[np.arange(monopoleCovariance.shape[0]), np.arange(quadrupoleCovariance.shape[1]), 2])
            self.dataQE = self.rawQE[selection]
        else:
            self.quadrupole = None
            self.rawQE = None
            self.dataQE = None
        
        
        
            
        
    def getData(self):
        return (self.dataX, self.monopole, self.dataE, self.quadrupole, self.dataQE, self.dataZ)
        
    def getRawData(self):
        return (self.rawX, self.rawMonopole, self.rawE, self.rawQuadrupole, self.rawQE, self.dataZ)
        
    
    def getModel(self, params, modelss):
        allParams = self.getParams()
        for i, p in enumerate(params):
            p = np.round(p, 5)
            if p <= allParams[i][1] or p >= allParams[i][2]:
                self.debug("Outside %s: %0.2f" % (allParams[i][0], p))
                return None
        cambParams = {k[0]: np.round(params[i],5) for (i,k) in enumerate(self.cambParams)}
        fitDict = {k[0]:params[i + len(cambParams)]  for (i,k) in enumerate(self.fitParams)}
        omch2 = cambParams.get('omch2')    
        
        if self.logkstar is None:
            if fitDict.get('sigmav') is not None:
                sigmav = fitDict['sigmav']
                kstar = 1 / (np.sqrt(2) * sigmav)
            else:
                kstar = np.exp(fitDict['kstar'])
        else:
            kstar = np.exp(self.logkstar)
        b2 = fitDict['b2']
        
        
        
        if self.generator is None:
            self.generator = methods.SlowGenerator(debug=True)
        (ks, pklin, pkratio) = self.generator.getOmch2AndZ(omch2, self.dataZ)
        
        pknw = methods.dewiggle(ks, pklin)

        
        weights = getLinearNoWiggleWeight(ks, kstar)
        pkdw = pklin * weights + pknw * (1 - weights)
        pknl = pkdw * pkratio
        mpknl = b2 * pknl
        
        if self.angular:
            beta = fitDict['beta']
            loren = fitDict['lorentzian']
            
            ksmu = ks[np.newaxis].T.dot(self.muNA )
            ar = mpknl[np.newaxis].T.dot(np.power((1 + beta * self.mu2NA), 2)) / (1 + (loren * loren * ksmu * ksmu))
        else:
            ar = mpknl
            
        s0 = 0.32
        gamma = -1.36
        
        
        if self.poles:
            alpha = fitDict['alpha']
            if self.matchQuad:
                qalpha = fitDict['qalpha']
            if self.angular:
                monopole = simps(ar, self.mu)
            else:
                monopole = mpknl
            datapointsM = methods.pk2xiGauss(ks, monopole, modelss * alpha)
            growthm = 1 + np.power(((modelss * alpha)/s0), gamma)
            datapointsM = datapointsM * growthm
            if self.matchQuad:        
                quadrupole = simps(ar * self.p2 * 5.0, self.mu)       
                datapointsQ = methods.pk2xiGaussQuad(ks, quadrupole, (modelss * qalpha))
                growthq = 1 + np.power(((modelss * qalpha)/s0), gamma)
                datapointsQ = datapointsQ * growthq
                return np.concatenate((datapointsM, datapointsQ))
            else:
                return datapointsM
                
        else:
            alphaPerp = fitDict['alphaPerp']
            alphaParallel = fitDict['alphaParallel']
            monopole = simps(ar, self.mu)
            quadrupole = simps(ar * self.p2 * 5, self.mu)
            hexapole = simps(ar * self.p4 * 9, self.mu)
            
            datapointsM = methods.pk2xiGauss(ks, monopole, self.ss) 
            datapointsQ = methods.pk2xiGaussQuad(ks, quadrupole, self.ss)
            datapointsH = methods.pk2xiGaussHex(ks, hexapole, self.ss)
            
            growth = 1 + np.power((self.ss/s0), gamma)

            datapointsM = datapointsM * growth
            datapointsQ = datapointsQ * growth
            datapointsH = datapointsH * growth
            
            sprime = modelss
            
            
            xi2dm = np.ones(self.mu.size)[np.newaxis].T.dot(datapointsM[np.newaxis])
            xi2dq = self.p2[np.newaxis].T.dot(datapointsQ[np.newaxis])
            xi2dh = self.p4[np.newaxis].T.dot(datapointsH[np.newaxis])
            xi2d = xi2dm + xi2dq + xi2dh

            mugrid = self.muNA.T.dot(np.ones((1, datapointsM.size)))
            ssgrid = self.ss[np.newaxis].T.dot(np.ones((1, self.mu.size))).T
            
            flatmu = mugrid.flatten()
            flatss = ssgrid.flatten()
            flatxi2d = xi2d.flatten()
            
            
            sqrtt = np.sqrt(alphaParallel * alphaParallel * self.mu2 + alphaPerp * alphaPerp * (1 - self.mu2))
            mus = alphaParallel * self.mu / sqrtt
            mu1 = self.mu[:self.mu.size/2]
            mu2 = self.mu[self.mu.size/2:]
            xiT = []
            xiL = []
            svals = np.array([])
            mvals = np.array([])
            for sp in sprime:
                svals = np.concatenate((svals, sp * sqrtt))
                mvals = np.concatenate((mvals, mus))
            
            xis = scipy.interpolate.griddata((flatmu, flatss), flatxi2d, (mvals, svals))
            for i, sp in enumerate(sprime):
                sz = sqrtt.size/2
                ii = 2 * i
                xis1 = xis[ii*sz : (ii+1)*sz]
                xis2 = xis[(ii+1)*sz : (ii+2)*sz]
                xiT.append(2 * simps(xis1, mu1))
                xiL.append(2 * simps(xis2, mu2))
            
            #return sqrtt, mvals, svals, xis, xiT, xiL, mugrid, ssgrid, xi2d, np.concatenate((np.array(xiT), np.array(xiL)))
            '''
            params = [0.12, 0.7, 0.7, -0.5, 2, 1, 1, 0]
            ss = np.linspace(1, 200, 100)
            w = WizColaCosmoMonopoleFitter()
            sqrtt, m,s,x,t,l,mugrid,sgrid,xi2d,model = w.getModel(params, ss)
            
            plt.plot(w.dataX, w.dataX*w.dataX*w.monopole, 'o', color='r')
            plt.plot(w.dataX, w.dataX*w.dataX*w.quadrupole, 'o', color='b')
            plt.plot(ss, ss*ss*model[:model.size/2], '--', color='r')
            plt.plot(ss, ss*ss*model[model.size/2:], '--', color='b')
            
            
            i = 52
            fig = plt.figure(figsize=(15, 11), dpi=150)
            c = plt.contourf(mugrid, sgrid, sgrid*sgrid*xi2d, 20)
            plt.scatter(m[i*sqrtt.size:(i+1)*sqrtt.size],s[i*sqrtt.size:(i+1)*sqrtt.size])
            cbar = plt.colorbar(c)
            
            fig = plt.figure(figsize=(13,11))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_wireframe(mugrid, sgrid, sgrid*sgrid*xi2d, rstride=1, cstride=1, color='k', alpha=0.3)
            ax.contourf(mugrid, sgrid, sgrid*sgrid*xi2d, 20, zdir='z', offset=-70)
            ax.contour(mugrid, sgrid, sgrid*sgrid*xi2d, 20, zdir='z', offset=80)
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.set_zlim(-70, 80)

            '''

            return np.concatenate((np.array(xiT), np.array(xiL)))
        
        
        
        

    def getChi2(self, params):
        datapoints = self.getModel(params, self.dataX)
        if datapoints is None:
            return None
        chi2 = np.dot((self.totalData - datapoints).T, np.dot(self.dataCov, self.totalData - datapoints))
        return chi2

class WizColaCosmoMonopoleFitter(CosmoMonopoleFitter):
    def __init__(self, cambFast=True, pk2xiFast=True, debug=True, bin=0, matchQuad=True, minS=50, maxS=150):
        super(self.__class__, self).__init__(cambFast=cambFast, pk2xiFast=pk2xiFast, debug=debug)
        
        wiz = WizColaLoader.getMultipoles()
        monopoles = wiz.getAllMonopoles(bin)
        quadrupoles = wiz.getAllQuadrupoles(bin)
        z = wiz.getZ(bin)
        datax = monopoles[:, 0]
        monopoles = monopoles[:, 1:]
        quadrupoles = quadrupoles[:, 1:]
        monopole = np.average(monopoles, axis=1)
        quadrupole = np.average(quadrupoles, axis=1)
        monopoleCor = wiz.getMonopoleCovariance(bin)
        quadrupoleCor = wiz.getQuadrupoleCovariance(bin)
        cor = wiz.getCovariance(bin)
        monopoleCor[:,:,3] *= np.sqrt(600)
        monopoleCor[:,:,2] /= np.sqrt(600)
        quadrupoleCor[:,:,3] *= np.sqrt(600)
        quadrupoleCor[:,:,2] /= np.sqrt(600)
        cor[:,:,3] *= np.sqrt(600)
        cor[:,:,2] /= np.sqrt(600)
        self.addCambParameter('omch2', 0.05, 0.25, '$\\Omega_c h^2$')
        self.setData(datax, monopole, quadrupole, cor, monopoleCor, quadrupoleCor, z, minS=minS, maxS=maxS, matchQuad=matchQuad)
     
class WizColaCosmoWedgeFitter(CosmoMonopoleFitter):
    def __init__(self, cambFast=False, pk2xiFast=False, debug=True, bin=0, minS=50, maxS=150):
        super(self.__class__, self).__init__(cambFast=cambFast, pk2xiFast=pk2xiFast, debug=debug)
        
        wiz = WizColaLoader.getWedges()
        trans = wiz.getAllTransverse(bin)
        longs = wiz.getAllLongitudinal(bin)
        z = wiz.getZ(bin)
        datax = trans[:, 0]
        trans = trans[:, 1:]
        longs = longs[:, 1:]
        tran = np.average(trans, axis=1)
        longss = np.average(longs, axis=1)
        transCor = wiz.getTransverseCovariance(bin)
        longCor = wiz.getLongitudinalCovariance(bin)
        cor = wiz.getCovariance(bin)
        transCor[:,:,3] *= np.sqrt(600)
        transCor[:,:,2] /= np.sqrt(600)
        longCor[:,:,3] *= np.sqrt(600)
        longCor[:,:,2] /= np.sqrt(600)
        cor[:,:,3] *= np.sqrt(600)
        cor[:,:,2] /= np.sqrt(600)
        self.addCambParameter('omch2', 0.05, 0.25, '$\\Omega_c h^2$')
        self.setData(datax, tran, longss, cor, transCor, longCor, z, minS=minS, maxS=maxS, poles=False)
        
class WigglezOldMonopoleFitter(CosmoMonopoleFitter):
    def __init__(self, cambFast=False, pk2xiFast=False, debug=True, bin=0, minS=10, maxS=180, angular=False, log=False):
        super(self.__class__, self).__init__(cambFast=cambFast, pk2xiFast=pk2xiFast, debug=debug)
        wig = WigglezOldLoader.getInstance()
        z = wig.getZ(bin)
        data = wig.getMonopoles(bin)
        cor = wig.getCov(bin)
        datax = data[:,0]
        monopole = data[:,1]
        print(monopole.size)
        self.addCambParameter('omch2', 0.05, 0.2, '$\\Omega_c h^2$')
        self.setData(datax, monopole, None, cor, cor, None, z, minS=minS, maxS=maxS, matchQuad=False, angular=angular, log=log)
        
if __name__ == '__main__':
    args = sys.argv
    
    params = [(0, 200), (20, 200), (40, 200), (20, 160), (20, 120), (40, 150), (40, 120), (10, 180), (40, 180), (25, 180)]
    if len(args) > 2: 
        a = int(args[2])
    else:
        a = 9
    if len(args) > 3:
        b = int(args[3])
    else:
        b = 1
    minS = params[a][0]
    maxS = params[a][1]
    finalFitter = WizColaCosmoMonopoleFitter(cambFast=False, pk2xiFast=False, bin=b, minS=minS, maxS=maxS, matchQuad=True)
    #finalFitter = WizColaCosmoWedgeFitter(bin=0, minS=minS, maxS=maxS)
    #finalFitter = WigglezOldMonopoleFitter(bin=b, minS=minS, maxS=maxS, angular=False, log=False)
    t = 'wizcola_quad'
    text = "%s_z%d_%d_%d" % (t, b, minS, maxS)
    print(text)
    
    numWalks = 4
    chunkLength = 10000
    maxs = 550000
    
    cambMCMCManager = CambMCMCManager(text, finalFitter, debug=True)
    cambMCMCManager.configureMCMC(numCalibrations=15,calibrationLength=1000, thinning=2, maxSteps=maxs)
    cambMCMCManager.configureSaving(stepsPerSave=1000)
    
    walk = None
    outputToFile = False
    if len(args) > 1:
        try:
            walk = int(args[1])
            outputToFile = False
        except Exception:
            print("Argument %s is not a number" % args[1])
    
    if walk is not None:
        if walk < 0:
            running = True
            currentWalk = 0
            while running:
                print("Running walk %d" % currentWalk)
                steps = cambMCMCManager.doWalk(currentWalk, outputToFile=outputToFile, chunkLength=chunkLength)    
                if currentWalk == numWalks - 1 and steps[:,0].size >= maxs:
                    running = False
                else:
                    currentWalk = (currentWalk + 1) % numWalks
        else:
            cambMCMCManager.doWalk(walk, outputToFile=outputToFile)
            
    else:
        cambMCMCManager.consolidateData()
        #cambMCMCManager.consolidateData(uid="wizcola_quad_z1_10_180")
        #cambMCMCManager.consolidateData(uid="wizcola_quad_z1_40_180")
        #cambMCMCManager.consolidateData(uid="wizcola_quad_z1_40_150")
        print(cambMCMCManager.getParameterBounds())
        print(cambMCMCManager.getBlakeParameterBounds())
        print(cambMCMCManager.getTamParameterBounds())
        cambMCMCManager.testConvergence()
        #cambMCMCManager.plotResults(filename=text)
        cambMCMCManager.plotResults()
        #cambMCMCManager.plotOmch2AndAlpha(filename="BcosmoComp", uids=['wizcola_quad_z1_40_180', 'wizcola_quad_z1_10_180','wizcola_quad_z1_40_150'], degeneracies=False, alpha=1.0, omch2=0.113, labels=["$40<s<180$", "$10 < s < 180$", "$40 < s < 150$"])

        #cambMCMCManager.plotMostLikelyModel(filename=text+'model')
        #cambMCMCManager.plotMostLikelyModel()
        #cambMCMCManager.plotWalk("omch2", "b2", final=False)
        #cambMCMCManager.plotWalk("omch2", "alphaPerp", final=False)
        #cambMCMCManager.plotWalk("omch2", "alphaParallel", final=False)
        #cambMCMCManager.plotWalk("alphaParallel", "alphaPerp", final=False)
        #cambMCMCManager.plotWalk("beta", "b2", final=False)
        #cambMCMCManager.plotWalk("lorentzian", "kstar", final=False)