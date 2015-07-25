import numpy as np


class tfFit2(object):
    def __init__(self, omega0, f_baryon, hubble, Tcmb=2.728):
        self.omega0 = omega0
        self.f_baryon = f_baryon
        self.hubble = hubble
        self.Tcmb = Tcmb


        omhh = omega0 * hubble * hubble
        
        self.setParams(omhh, f_baryon, Tcmb)
        
        
    def __call__(self, ks):
        tf_full = np.zeros(ks.size)        
        tf_baryon = np.zeros(ks.size)        
        tf_cdm = np.zeros(ks.size)        
        tf_nw = np.zeros(ks.size)        
        tf_zeroBaryon = np.zeros(ks.size)        
        for i, k in enumerate(ks):
            (full, baryon, cdm) = self.tfTransferFunction(k * self.hubble)
            nw = self.tfNoWiggles(k * self.hubble)
            zb = self.TF_zerobaryon(k * self.hubble / self.omhh * (self.Tcmb/2.7)**2)
            tf_full[i] = full
            tf_baryon[i] = baryon
            tf_cdm[i] = cdm
            tf_nw[i] = nw
            tf_zeroBaryon[i] = zb
            
        return {
            'ks': ks,
            'full': tf_full,
            'baryon': tf_baryon,
            'cdm': tf_cdm,
            'nw': tf_nw,
            'zb': tf_zeroBaryon
        }
        
        
    def tfNoWiggles(self, k):
        a = self.alpha_gamma(self.omhh, self.f_baryon)
        q_eff = k / self.omhh * (self.Tcmb/2.7)**2
        q_eff = q_eff / (a + (1.0 - a)/ (1.0 + (0.43 * k * self.sound_horizon_fit(self.omhh, self.f_baryon))**4))

        TF_nowiggles = self.TF_zerobaryon(q_eff)  
        return TF_nowiggles      
        
    def alpha_gamma(self, omhh, f_baryon):
        alpha_gamma = 1.0 - 0.328 * np.log(431.0 * omhh) * f_baryon + 0.38 * np.log(22.3 * omhh) * (f_baryon)**2
        return alpha_gamma
        
    def sound_horizon_fit(self, omhh, f_baryon):
        sound_horizon_fit = 44.5 * np.log(9.83 / omhh) / np.sqrt(1.0 + 10.0 * self.obhh**(0.75))
        return sound_horizon_fit
        
    def TF_zerobaryon(self, q):
        TF_zerobaryon = np.log(2.0 * np.exp(1.0) + 1.8 * q)
        TF_zerobaryon = TF_zerobaryon / (TF_zerobaryon + (14.2 + 731.0/(1 + 62.5 * q)) * q**2)
        return TF_zerobaryon
        
    def tfTransferFunction(self, k):
        if (k < 0): 
            raise Exception("k is negative. Why you do this")
            
        q = k / 13.41 / self.k_equality
        ks = k * self.sound_horizon
           
        tf_cdm = 1.0 / (1.0 + (ks / 5.4)**4.0)
        tf_cdm = tf_cdm * self.TF_pressureless(q, 1.0, self.beta_c) + (1.0 - tf_cdm) * self.TF_pressureless(q, self.alpha_c, self.beta_c)
        s_tilde = self.sound_horizon / (1.0 + (self.beta_node / ks)**3.0)**(1./3.) 
        tf_baryon = self.TF_pressureless(q, 1., 1.) / (1. + (ks / 5.2)**2.)
        tf_baryon = tf_baryon + self.alpha_b / (1. + (self.beta_b / ks)**3) * np.exp(-(k / self.k_silk)**(1.4))
        tf_baryon = tf_baryon *(np.sin(k * s_tilde) / (k * s_tilde))
        tf_full = self.f_baryon * tf_baryon + (1 - self.f_baryon) * tf_cdm
        
        return (tf_full, tf_baryon, tf_cdm)
        
    def TF_pressureless(self, q, a, b):
        TF_pressureless = np.log(np.exp(1.0) + 1.8 * b * q)
        TF_pressureless = TF_pressureless / (TF_pressureless + (14.2 / a + 386/(1.0 + 69.9 * q**1.08)) * q**2)
        return TF_pressureless
        
    def setParams(self, omhh, f_baryon, Tcmb):
        '''
        self.theta_cmb = -1.0
        self.z_equality = -1.0
        self.k_equality = -1.0
        self.z_drag = -1.0
        self.R_drag = -1.0
        self.R_equality = -1.0
        self.sound_horizon = -1.0
        self.k_silk = -1.0
        self.alpha_c = -1.0
        self.beta_c = -1.0
        self.alpha_b = -1.0
        self.beta_b = -1.0
        self.beta_node = -1.0 '''
        self.omhh = omhh

        if (f_baryon < 0):
            self.f_baryon = 1.0e-5
        if (Tcmb < 0):
            self.Tcmb = Tcmb
        if (self.omhh < 0):
            raise Exception("omhh is negative. Dont do that.")
        if (self.hubble > 10):
            print "Wanring, hubble is in number of 100 km/s/Mpc, so 0.7 gives 70km/s/Mpc"
        
        # Auxiliary variables
        self.obhh = omhh * f_baryon
        self.theta_cmb = Tcmb / 2.7

        # Main variables
        self.z_equality = 2.50e4 * self.omhh * self.theta_cmb**(-4.0) - 1.0
        self.k_equality = 0.0746 * self.omhh * self.theta_cmb**(-2.0) 

        self.z_drag = 0.313 * self.omhh**(-0.419) * (1.0 + 0.607 * self.omhh**(0.674))
        self.z_drag = 1e0 + self.z_drag * self.obhh**(0.238 * self.omhh**(0.223))
        self.z_drag = 1291e0 * self.omhh**(0.251) / (1e0 + 0.659 * self.omhh**(0.828)) * self.z_drag
 
        self.R_drag = 31.5 * self.obhh * self.theta_cmb**(-4.0) * 1000e0 / (1e0 + self.z_drag) 
        self.R_equality = 31.5 * self.obhh * self.theta_cmb**(-4.0) * 1000e0 / (1e0 + self.z_equality) 

        self.sound_horizon = 2.0 / 3.0 / self.k_equality * np.sqrt(6.0 / self.R_equality) * \
                             np.log(( np.sqrt(1.0 + self.R_drag) + np.sqrt(self.R_drag + self.R_equality) ) / \
                             (1.0 + np.sqrt(self.R_equality)))

        self.k_silk = 1.6 * self.obhh**(0.52) * self.omhh**(0.73) * (1e0 + (10.4 * self.omhh)**(-0.95))

        self.alpha_c = ((46.9 * self.omhh)**(0.670)*(1e0+(32.1 * self.omhh)**(-0.532)))
        self.alpha_c = self.alpha_c**(-f_baryon) 
        self.alpha_c = self.alpha_c * ((12.0 * self.omhh)**(0.424)*(1e0 + (45.0 * self.omhh)**(-0.582)))**(-f_baryon**3.0)

    
        self.beta_c = 0.944 / (1 + (458.0 * self.omhh)**(-0.708))
        self.beta_c = 1.0 + self.beta_c * ((1.0 - f_baryon)**((0.395 * self.omhh)**(-0.0266)) - 1e0)
        self.beta_c = 1.0 / self.beta_c

        y = (1e0 + self.z_equality) / (1e0 + self.z_drag)
        self.alpha_b = y * (-6.0 * np.sqrt(1.0 + y) + (2.0 + 3.0 * y) * np.log((np.sqrt(1.0 + y) + 1.0) / (np.sqrt(1.0 + y) - 1.0)))
        self.alpha_b = 2.07 * self.k_equality * self.sound_horizon * (1.0 + self.R_drag)**(-0.75) * self.alpha_b


        self.beta_b = 0.5 + f_baryon + (3.0 - 2.0 * f_baryon) *  np.sqrt((17.2 * self.omhh)**2.0 + 1e0)

        self.beta_node = 8.41 * self.omhh**(0.435)


        
































