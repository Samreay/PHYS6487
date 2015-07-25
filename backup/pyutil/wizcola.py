import numpy as np
import os
import re

class WizColaLoader(object):
    @staticmethod
    def getMultipoles():
        return WizColaMultipoles()
        
    @staticmethod
    def getWedges():
        return WizColaWedges()

class WizCola(object):
    def __init__(self, path):
        thisDir, thisFilename = os.path.split(__file__)
        self.path = thisDir + os.sep + path
        self.z1Pattern = re.compile('_z([0-9]+pt[0-9]+)_')
        self.z2Pattern = re.compile('_z.*_([0-9]+pt[0-9]+)\.')
        self.zs = [0.44, 0.6, 0.73]
        
        self.files = []
        self.bins = []
        self.covs = []
        self.binz1 = []
        self.binz2 = []
                
        for f in self.getDatFilesInDir(self.path):
            self.loadFile(self.path, f)
        
        for f in self.getDatFilesInDir(self.path + os.sep + 'cov'):
            self.loadCovariance(self.path + os.sep + 'cov', f)
        
    def loadCovariance(self, path, filename):
        a = np.loadtxt(path + os.sep + filename)
        self.covs.append(a.reshape((60,60,5)))
        
    def getCorner(self, bin, corner=0):
        a = self.covs[bin]
        tmp = a[a[:,:,4] == corner]
        if corner == 1:
            tmp = tmp[:tmp[:,0].size / 2, :]
        return tmp[:,:4].reshape((30,30,4))

    def getCovariance(self, bin):
        return self.covs[bin]
        
    def getCrossCovariance(self, bin):
        return self.getCorner(bin, corner=1)
        
    def getZ(self, bin):
        return self.zs[bin]
        
    def getZ1FromName(self, filename):
        z1String = self.z1Pattern.search(filename)
        if z1String:
            return float(z1String.group(1).replace('pt', '.'))
        else:
            return None
                
    def getZ2FromName(self, filename):
        z2String = self.z2Pattern.search(filename)
        if z2String:
            return float(z2String.group(1).replace('pt', '.'))
        else:
            return None
         
    def getDatFilesInDir(self, dir):
        return [i for i in os.listdir(dir) if i.endswith('.dat')]
    
    def loadFile(self, path, filename):
        a = np.loadtxt(path + os.sep + filename)
        self.files.append(filename)
        self.bins.append(a)
        self.binz1.append(self.getZ1FromName(filename))
        self.binz2.append(self.getZ2FromName(filename))
        

class WizColaMultipoles(WizCola):
    def __init__(self):
        super(WizColaMultipoles, self).__init__('wizcola' + os.sep + 'multipoles')
        
    def getMonopoles(self, bin):
        a = self.bins[bin]
        height = a[:,0].size / 2
        return self.bins[bin][:height,:]
    

    def getAllMonopoles(self, bin):
        a = self.bins[bin]
        height = a[:,0].size / 2
        b = self.bins[bin][:height,:]
        return b[:, 2:]
		
    def getMonopoleSimulation(self, bin, simNum=0):
        return self.getMonopoles(bin)[:, [2, 3 + simNum]]
        
    def getQuadrupoles(self, bin):
        a = self.bins[bin]
        height = a[:,0].size / 2
        return self.bins[bin][height:,:]
    
    def getAllQuadrupoles(self, bin):
        a = self.bins[bin]
        height = a[:,0].size / 2
        b = self.bins[bin][height:,:]
        return b[:, 2:]
		
    def getMonopoleCovariance(self, bin):
        return self.getCorner(bin, 0)
        
    def getQuadrupoleCovariance(self, bin):
        return self.getCorner(bin, 2)
    
class WizColaWedges(WizCola):
    def __init__(self):
        super(WizColaWedges, self).__init__('wizcola' + os.sep + 'wedges')
    
    def getTransverse(self, bin):
        a = self.bins[bin]
        height = a[:,0].size / 2
        return self.bins[bin][:height,:]
        
    def getAllTransverse(self, bin):
        a = self.bins[bin]
        height = a[:,0].size / 2
        b = self.bins[bin][:height,:]
        return b[:, 2:]
        
    def getAllLongitudinal(self, bin):
        a = self.bins[bin]
        height = a[:,0].size / 2
        b = self.bins[bin][height:,:]
        return b[:, 2:]
        
    def getTransverseSimulation(self, bin, simNum=0):
        return self.getTransverse(bin)[:, [2, 3 + simNum]]        
        
    def getTransverseCovariance(self, bin):
        return self.getCorner(bin, 0)
        
    def getLongitudinalCovariance(self, bin):
        return self.getCorner(bin, 2)
        
    