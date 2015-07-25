import numpy as np

class SmithCorr(object):
    
    def __call__(self, om, fb, hh, sig8, ns, zz, bb, klinlog, kmin, kmax, nkmod):
        
        #maxnkmod = 10000
        ol = 1.0 - om
        ob = fb * om
        temp = ob * (1.0 + np.sqrt(2.0 * hh) / om)
        gamm = om * hh * np.exp(-temp)
        
        pkcorr = np.zeros(nkmod)
        kkhalo = np.zeros(nkmod)
        pklin = np.zeros(nkmod)
        pknl = np.zeros(nkmod)
        # This should be called with nkmod = maxnkmod I believe
        self.getHaloFormula(om, ol, sig8, ns, gamm, zz, bb, kmin, kmax, nkmod, klinlog, kkhalo, pklin, pknl)
        
        for i in range(nkmod):
            pkcorr[i] = pknl[i] / pklin[i]
            
            
        return (kkhalo, pkcorr)
        
        
    #Peacock & Smith code.                                                *
    #                                                                      *
    # Inputs:                                                              *
    #                                                                      *
    # om_m0 = input z=0 matter density                                     *
    # om_v0 = input z=0 vacuum density                                     *
    # sig8  = input z=0 normalisation of cdm power spectrum                *
    # ns    = input primordial spectral index                              *
    # gams  = input shape parameter of cdm power spectrum (gams)           *
    # z     = input required redshift (z)                                  *
    # b     = linear bias factor     
    def getHaloFormula(self, inom, inol, insig8, inns, ingams, inz, bb, kmin, kmax, ndat, linlog, kkhalo, pklinhalo, pknlhalo):
        self.om_m = -1.0
        self.om_v = -1.0
        self.om_b = -1.0
        self.h = -1.0
        self.p_index = -1.0
        self.gams = -1.0
        self.sig8 = -1.0
        self.amp = -1.0
        
        om_m0 = inom
        om_v0 = inol
        self.sig8 = insig8
        self.gams = ingams
        self.p_index = inns
        self.z = inz
        
        aexp = 1./(1. + self.z) # expansion factor
        
        if (linlog == 2):
            lkmin = np.log10(kmin)
            lkmax = np.log10(kmax)
        
        # calculate matter density, vacuum density at desired redshift
        
        self.om_m = self.getOmegaM(aexp, om_m0, om_v0)
        self.om_v = self.getOmegaV(aexp, om_m0, om_v0)
        
        # calculate the amplitude of the power spectrum at desired redshift
        # using linear growth factors

        grow = self.gg2(self.om_m, self.om_v) 
        grow0 = self.gg2(om_m0, om_v0)       
        self.amp = aexp * bb * (grow / grow0)
        
        
        # calculate nonlinear wavenumber (rknl), effective spectral index (rneff)
        # and curvature (rncur) of the power spectrum at the desired redshift,
        # using method described in Smith et al (2002).

        # Almost about to start ranting at the code right now
        xlogr1 = -2.0
        xlogr2 = 3.5
        epsilon = 0.001
        while (True):
            rmid = (xlogr2 + xlogr1) / 2.0
            rmid = np.power(10, rmid)
            (sig, d1, d2) = self.wint(rmid)
            diff = sig - 1.0
            if (abs(diff) < epsilon):
                rknl = 1.0 / rmid
                rneff = -3 - d1
                rncur = -d2
                break
            elif (diff > epsilon):
                xlogr1 = np.log10(rmid)
            elif (diff < -epsilon):
                xlogr2 = np.log10(rmid)
            else:
                raise Exception("The infinite while loop did not converge: %0.3f" % diff)
        
        
        # now calculate power spectra for a range of wavenumbers (rk)
        for i in range(ndat):
            if (linlog == 1):
                rk = kmin + ((kmax - kmin) * ((1.0 * (i+1) - 0.5) / (1.0 * ndat)))
            else:
                rk = lkmin + ((lkmax - lkmin) * (((1.0 * (i+1)) - 1) / (1.0 * ndat - 1)))
                rk = 10**rk
                                
            kkhalo[i] = rk
                    
            # linear power spectrum !! Remeber => plin = k^3 * P(k) * constant
            # constant = 4*pi*V/(2*pi)^3 
            
            plin = self.amp * self.amp * self.p_cdm(rk)

            # calculate nonlinear power according to halofit: pnl = pq + ph,
            # where pq represents the quasi-linear (halo-halo) power and 
            # where ph is represents the self-correlation halo term. 

            (pnl, pq, ph) = self.halofit(rk, rneff, rncur, rknl, plin)
            
            pklinhalo[i] = plin
            pknlhalo[i] = pnl
            
        # comparison with Peacock & Dodds (1996)
        # THIS DOES NOT APPEAR TO BE USED IN THE CODE
        '''
        for i in range(ndat):
            if (linlog == 1):
                rklin = kmin + ((kmax - kmin) * (((1.0*i) - 0.5) / (1.0 * ndat)))
            else:
                rklin = lkmin + ((lkmax - lkmin) * (((i*1.0) - 1) / ((1.0 * ndat) - 1)))
                rklin = 10**rklin
                
            plin = self.amp * self.amp * self.p_cdm(rklin)
            
            # effective spectral index: rn_pd=dlogP(k/2)/dlogk
            rn_pd = rn_cdm(rklin)
    
            # nonlinear power from linear power
            pnl_pd = f_pd(plin, rn_pd)    
            
            rk = rklin * (1 + pnl_pd)**(1./3.)
        '''
    
    def f_pd(self, y, rn):
        g = (5./2.)*self.om_m/(self.om_m**(4./7.)-self.om_v+(1+self.om_m/2.)*(1+self.om_v/70.))
        a = 0.482*(1.+rn/3.)**(-0.947)
        b = 0.226*(1.+rn/3.)**(-1.778)
        alp = 3.310*(1.+rn/3.)**(-0.244)
        bet = 0.862*(1.+rn/3.)**(-0.287)
        vir = 11.55*(1.+rn/3.)**(-0.423)
        f_pd = y * ( (1.+ b*y*bet + (a*y)**(alp*bet)) / (1.+ ((a*y)**alp*g*g*g/vir/y**0.5)**bet ) )**(1./bet) 
        return f_pd
        
        
    def rn_cdm(self, rk):
        y = self.p_cdm(rk/2.0)
        yplus = self.p_cdm(rk*1.01/2.0)
        rn_cdm = -3.0 + np.log(yplus / y) * 100.5
        return rn_cdm
        
    # halo model nonlinear fitting formula as described in 
    # Appendix C of Smith et al. (2002)
    def halofit(self, rk, rn, rncur, rknl, plin):
        gam = 0.86485 + 0.2989 * rn + 0.1631 * rncur
        a = 1.4861 + 1.83693 * rn + 1.67618 * rn * rn + 0.7940 * rn * rn * rn+ 0.1670756 * rn * rn * rn * rn - 0.620695 * rncur
        a = 10**a      
        b = 10**(0.9463 + 0.9466 * rn + 0.3084 * rn * rn - 0.940 * rncur)
        c = 10**(-0.2807 + 0.6669 * rn + 0.3214 * rn * rn - 0.0793 * rncur)
        xmu = 10**(-3.54419 + 0.19086 * rn)
        xnu = 10**(0.95897 + 1.2857 * rn)
        alpha = 1.38848+0.3701 * rn - 0.1452 * rn * rn
        beta = 0.8291 + 0.9854 * rn + 0.3400 * rn**2

        if (abs(1 - self.om_m) > 0.01): # omega evolution 
            f1a = self.om_m**(-0.0732)
            f2a = self.om_m**(-0.1423)
            f3a = self.om_m**(0.0725)
            f1b = self.om_m**(-0.0307)
            f2b = self.om_m**(-0.0585)
            f3b = self.om_m**(0.0743)       
            frac = self.om_v / (1. - self.om_m) 
            f1 = frac * f1b + (1 - frac) * f1a
            f2 = frac * f2b + (1 - frac) * f2a
            f3 = frac * f3b + (1 - frac) * f3a
        else:  
            f1=1.0
            f2=1.0
            f3=1.0
      

        y = (rk / rknl)
        ph = a * y**(f1 * 3) / (1 + b * y**(f2) + (f3 * c * y)**(3 - gam))
        ph = ph / (1 + xmu * y**(-1) + xnu * y**(-2))
        pq = plin*(1 + plin)**beta / (1 + plin * alpha) * np.exp(-y / 4.0 - (y**2) / 8.0)
        
        pnl = pq + ph
        
        return (pnl, pq, ph)

                
    # Bond & Efstathiou (1984) approximation to the linear CDM power spectrum     
    def p_cdm(self, rk):
        rkeff = 0.172 + 0.011 * np.log(self.gams / 0.36) * np.log(self.gams / 0.36)
        q = 1.0e-20 + rk / self.gams
        q8 = 1.0e-20 + rkeff / self.gams
        tk = 1 / (1 + (6.4 * q + np.power((3.0 * q), 1.5) + (1.7 * q)**2)**1.13)**(1 / 1.13)
        tk8 = 1 / (1 + (6.4 * q8 + (3.0 * q8)**1.5 + (1.7 * q8)**2)**1.13)**(1/1.13)
        p_cdm = self.sig8 * self.sig8 * ((q / q8)**(3. + self.p_index)) * tk * tk / tk8 / tk8
        return p_cdm
        
    # The subroutine wint, finds the effective spectral quantities
    # rknl, rneff & rncur. This it does by calculating the radius of 
    # the Gaussian filter at which the variance is unity = rknl.
    # rneff is defined as the first derivative of the variance, calculated 
    # at the nonlinear wavenumber and similarly the rncur is the second
    # derivative at the nonlinear wavenumber. 
    def wint(self, r):
        nint = 3000
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        for i in range(nint):
            t = ((1.0 * (i+1)) - 0.5) / (nint)
            y = -1.0 + 1.0 / t
            rk = y
            d2 = self.amp * self.amp * self.p_cdm(rk)
            x = y * r
            w1 = np.exp(-x * x)
            w2 = 2 * x * x * w1
            w3 = 4 * x * x * (1 - x * x) * w1
            sum1 += w1 * d2 / y / t / t
            sum2 += w2 * d2 / y / t / t
            sum3 += w3 * d2 / y / t / t
        sum1 /= nint
        sum2 /= nint
        sum3 /= nint
        sig = np.sqrt(sum1)
        d1 = -sum2 / sum1
        d2 = - sum2 * sum2 / sum1 / sum1 - sum3 / sum1

        return (sig, d1, d2)
        
    # evolution of omega matter with expansion factor
    def getOmegaM(self, aa, om_m0, om_v0):
        omega_t = 1.0 + (om_m0 + om_v0 -1.0) / (1 - om_m0 - om_v0 + om_v0 * aa * aa + om_m0/aa)
        omega_m = omega_t * om_m0 / (om_m0 + om_v0 * aa * aa * aa)
        return omega_m
    
    # evolution of omega lambda with expansion factor
    def getOmegaV(self, aa, om_m0, om_v0):
        omega_t = 1.0 + (om_m0 + om_v0 - 1.0) / (1 - om_m0 - om_v0 + om_v0 * aa * aa + om_m0/aa)
        omega_v = omega_t * om_v0 / (om_v0 + om_m0 / aa / aa / aa)
        return omega_v
    
    # growth factor for linear fluctuations 
    def gg2(self, om_m, om_v):
        val = 2.5 * om_m / (np.power(om_m, (4./7.)) - om_v + (1.0 + om_m / 2.) * (1. + om_v / 70.))
        return val
        