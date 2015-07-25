import numpy as np
import os
import re

class WigglezOldLoader(object):
    @staticmethod
    def getInstance():
        return WigglezOld()

class WigglezOld(object):
    def __init__(self, path='wigglezold'):
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


    def getCovariance(self, bin):
        return self.covs[bin]
        
        
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
        ndatapoints = a[0,0]
        data = a[1:1+ndatapoints,0:2]
        cov = a[1 + ndatapoints:,:].reshape((ndatapoints,ndatapoints,3))
        cor = np.linalg.inv(cov[:,:,2])
        cov = np.dstack((cov[:,:,0], cov[:,:,1], cov[:,:,2], cor))
        self.bins.append(data)
        self.covs.append(cov)
        self.binz1.append(self.getZ1FromName(filename))
        self.binz2.append(self.getZ2FromName(filename))
        
    def getMonopoles(self, bin):
        return self.bins[bin]
        
    def getCov(self, bin):
        return self.covs[bin]
        