from sam import *
import os
import matplotlib, matplotlib.pyplot as plt

thisDir, thisFilename = os.path.split(__file__)
path = thisDir + os.sep + 'test'

def testTfFitNW():
    results = np.loadtxt(path + os.sep + 'tffit.txt')
    
    ks = results[:,0]
    T = results[:,1]
    Tb = results[:,2]
    Tc = results[:,3]
    Tnw = results[:,4]
    Tzb = results[:,5]
    
    om = 0.273829
    ox = 0.726171
    oc = 0.228158
    ok = 0.00000
    ob = 0.0456717
    hh = 0.705000
    s8 = 0.800000
    amp = 2.48700
    ns = 0.960000
    zz = 0.600000
    massless = 3.04600
    massive = 0.00000
    onuh2 = 0.00000
    
    tt = tfFit2(om, ob/om, hh, Tcmb=2.725)
    res = tt(ks)
    
    Tdelta = 100*(res['nw'] - Tnw) / Tnw
    
    
    fig = plt.figure(figsize=(14,7), dpi=300)
    matplotlib.rcParams.update({'font.size': 14})
    ax0 = fig.add_subplot(2,1,1)
    
    ax0.plot(ks, ks * ks * ks**ns * Tnw * Tnw, color='b', label='Actual', alpha=0.7)
    ax0.plot(ks, ks * ks * ks**ns * res['nw'] * res['nw'], color='r', label='Python', alpha=0.7)
    ax0.set_xscale('log')
    ax1 = fig.add_subplot(2,1,2)
    ax1.plot(ks, Tdelta, color='r', label='delta')
    ax1.set_xscale('log')

def testTfFitFull():
    results = np.loadtxt(path + os.sep + 'tffit.txt')
    
    ks = results[:,0]
    T = results[:,1]
    Tb = results[:,2]
    Tc = results[:,3]
    Tnw = results[:,4]
    Tzb = results[:,5]
    
    om = 0.273829
    ox = 0.726171
    oc = 0.228158
    ok = 0.00000
    ob = 0.0456717
    hh = 0.705000
    s8 = 0.800000
    amp = 2.48700
    ns = 0.960000
    zz = 0.600000
    massless = 3.04600
    massive = 0.00000
    onuh2 = 0.00000
    
    tt = tfFit2(om, ob/om, hh, Tcmb=2.725)
    res = tt(ks)
    
    Tdelta = 100*(res['full'] - T) / T
    
    
    fig = plt.figure(figsize=(14,7), dpi=300)
    matplotlib.rcParams.update({'font.size': 14})
    ax0 = fig.add_subplot(2,1,1)
    
    ax0.plot(ks, ks * ks * ks**ns * T * T, color='b', label='Actual', alpha=0.7)
    ax0.plot(ks, ks * ks * ks**ns * res['full'] * res['full'], color='r', label='Python', alpha=0.7)
    ax0.set_xscale('log')
    ax1 = fig.add_subplot(2,1,2)
    ax1.plot(ks, Tdelta, color='g', label='delta')
    ax1.set_xscale('log')

def testSmithCorrection():
    results = np.loadtxt(path + os.sep + 'smith_test.txt')

    ks = results[:, 0]
    r = results[:, 1]
    om = 0.273829
    ox = 0.726171
    oc = 0.228158
    ok = 0.00000
    ob = 0.0456717
    hh = 0.705000
    s8 = 0.800000
    amp = 2.48700
    ns = 0.960000
    zz = 0.600000
    massless = 3.04600
    massive = 0.00000
    onuh2 = 0.00000
    bb = 1
    
    (karr, pkratio) = SmithCorr()(om, ob/om, hh, sig8, ns, zz, bb, 2, 0.00100000, 40, 5000)
    
    rDelta = 100*(pkratio - r) / r
    
    
    fig = plt.figure(figsize=(14,7), dpi=300)
    matplotlib.rcParams.update({'font.size': 14})
    ax0 = fig.add_subplot(2,1,1)
    
    ax0.plot(ks, r, color='b', label='Actual', alpha=0.7)
    ax0.plot(ks, pkratio, color='r', label='Python', alpha=0.7)
    ax0.set_xscale('log')
    ax1 = fig.add_subplot(2,1,2)
    ax1.plot(ks, rDelta, color='g', label='delta')
    ax1.set_xscale('log')
