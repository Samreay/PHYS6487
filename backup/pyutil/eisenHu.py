import numpy as np

class tfFit(object):
    """ Python implementation of the algorithm developed by Eisenstein and Hu in 1997.
    Example code to use this class:
    
    ks = np.logspace(-2, 2.2, 1000)
    a = tfFit(0.3, 0.5, 70)
    b = a(ks)
    """

    
    def __init__(self, omega0, fBaryon, hubble, Tcmb=2.728):
        
        
        # Just doing some variable checking
        if type(omega0)  is not float: self.error('Omega0 parameter is not a float, it is %s' % type(omega0))
        if type(fBaryon) is not float: self.error('fBaryon parameter is not a float, it is %s' % type(fBaryon))
        if type(hubble)  is not float and type(hubble) is not int: self.error('hubble parameter is not a float, it is %s' % type(hubble))
        if type(Tcmb)    is not float: self.error('Tcmb parameter is not a float, it is %s' % type(Tcmb))
            
        self.omega0 = omega0
        self.fBaryon = fBaryon
        self.hubble = hubble * 1.0
        self.Tcmb = Tcmb
        
        self.setParameters(omega0 * hubble * hubble, fBaryon, Tcmb)
        

        
    def __call__(self, ks):
        """ Remember: ks is in units of h Mpc^-1. """

        if type(ks)  is not np.ndarray: self.error('ks parameter is not type numpy.ndarray, it is %s' % type(ks))
        tf_full = np.zeros(ks.size)
        tf_baryon = np.zeros(ks.size)
        tf_cdm = np.zeros(ks.size)
        
        for i, k in enumerate(ks):
            v, b, c = self.fit_onek(k * self.hubble)
            tf_full[i] = v
            tf_baryon[i] = b
            tf_cdm[i] = c
            
        return {
                'ks': ks,
                'full': tf_full,
                'baryon': tf_baryon,
                'cdm': tf_cdm
                }
        
    def error(self, message):
        raise Exception(message)
        
    def setParameters(self, omega0hh, fBaryon, Tcmb):
        
        # Set global variables
        self.omhh = -1.0              # Omega_matter*h^2 
        self.obhh = -1.0              # Omega_baryon*h^2 
        self.theta_cmb = -1.0         # Tcmb in units of 2.7 K 
        self.z_equality = -1.0        # Redshift of matter-radiation equality, really 1+z 
        self.k_equality = -1.0        # Scale of equality, in Mpc^-1 
        self.z_drag = -1.0            # Redshift of drag epoch 
        self.R_drag = -1.0            # Photon-baryon ratio at drag epoch 
        self.R_equality = -1.0        # Photon-baryon ratio at equality epoch 
        self.sound_horizon = -1.0     # Sound horizon at drag epoch, in Mpc 
        self.k_silk = -1.0            # Silk damping scale, in Mpc^-1 
        self.alpha_c = -1.0           # CDM suppression 
        self.beta_c = -1.0            # CDM log shift 
        self.alpha_b = -1.0           # Baryon suppression 
        self.beta_b = -1.0            # Baryon envelope shift 
        self.beta_node = -1.0         # Sound horizon shift
        self.k_peak = -1.0            #Fit to wavenumber of first peak, in Mpc^-1 
        self.sound_horizon_fit = -1.0 # Fit to sound horizon, in Mpc 
        self.alpha_gamma = -1.0
        
        if (fBaryon < 0.0):
            self.error("fBaryon cannot be negative")
        if (omega0hh < 0.0):
            self.error("omega0hh cannot be negative")
        if (Tcmb < 0.0):
            self.error("Tcmb cannot be negative")
        
        self.theta_cmb = Tcmb / 2.7
        self.omhh = omega0hh
        self.obhh = self.omhh * fBaryon
        
        self.z_equality = 2.50e4 * self.omhh / np.power(self.theta_cmb, 4)
        self.k_equality = 0.0746 * self.omhh / np.sqrt(self.theta_cmb)
        
        zDragB1 = 0.313 * np.power(self.omhh, -0.419) * (1 + 0.607 * np.power(self.omhh, 0.674))
        zDragB2 = 0.238 * np.power(self.omhh, 0.223)
        self.z_drag = 1291 * np.power(self.omhh, 0.251) / (1 + 0.659 * np.power(self.omhh, 0.828)) * (1 + zDragB1 * np.power(self.obhh, zDragB2))
        
        self.R_drag = 31.5 * self.obhh / np.power(self.theta_cmb, 4) * (1000 / (1 + self.z_drag))
        self.R_equality = 31.5 * self.obhh / np.power(self.theta_cmb, 4) * (1000 / self.z_equality)
        
        
        self.sound_horizon = 2.0 / 3.0 / self.k_equality * np.sqrt(6.0 / self.R_equality) * np.log((np.sqrt(1 + self.R_drag) + np.sqrt(self.R_drag + self.R_equality)) / (1 + np.sqrt(self.R_equality)))
        
        self.k_silk = 1.6 * np.power(self.obhh, 0.52) * np.power(self.omhh, 0.73) * (1 + np.power(10.4 * self.omhh, -0.95))
    
        self.alpha_c_a1 = np.power(46.9 * self.omhh, 0.670) * (1 + np.power(32.1 * self.omhh, -0.532))
        self.alpha_c_a2 = np.power(12.0 * self.omhh, 0.424) * (1 + np.power(45.0 * self.omhh, -0.582))
        self.alpha_c = np.power(self.alpha_c_a1, -fBaryon) * np.power(self.alpha_c_a2, -np.power(fBaryon, 3))
        
        
        self.beta_c_b1 = 0.944 / (1 + np.power(458 * self.omhh, -0.708))
        self.beta_c_b2 = np.power(0.395 * self.omhh, -0.0266)
        self.beta_c = 1.0 / (1 + self.beta_c_b1 * (np.power(1 - fBaryon, self.beta_c_b2) - 1))
    
        self.y = self.z_equality / (1 + self.z_drag)
        self.alpha_b_G = self.y * (-6. * np.sqrt(1 + self.y) + (2. + 3. * self.y) * np.log((np.sqrt(1 + self.y) + 1) / (np.sqrt(1 + self.y) - 1)))
        self.alpha_b = 2.07 * self.k_equality * self.sound_horizon * np.power(1 + self.R_drag, -0.75) * self.alpha_b_G
    
        self.beta_node = 8.41 * np.power(self.omhh, 0.435)
        self.beta_b = 0.5 + fBaryon + (3. - 2. * fBaryon) * np.sqrt(np.power(17.2 * self.omhh, 2.0) + 1)
    
        self.k_peak = 2.5 * 3.14159 * (1 + 0.217 * self.omhh) / self.sound_horizon
        self.sound_horizon_fit = 44.5 * np.log(9.83 / self.omhh) / np.sqrt(1 + 10.0 * np.power(self.obhh, 0.75))
    
        self.alpha_gamma = 1 - 0.328 * np.log(431.0 * self.omhh) * fBaryon + 0.38 * np.log(22.3 * self.omhh) * np.power(fBaryon, 2);
        
        
        
        
    def fit_onek(self, k):
        
        k = np.abs(k)
        if k == 0.0:
            return (1.0, 1.0, 1.0)
        
        q = k / 13.41 / self.k_equality
        xx = k * self.sound_horizon
        
        T_c_ln_beta = np.log(2.718282+1.8 * self.beta_c * q)
        T_c_ln_nobeta = np.log(2.718282+1.8 * q)
        T_c_C_alpha = 14.2 / self.alpha_c + 386.0 / (1+69.9 * np.power(q, 1.08))
        T_c_C_noalpha = 14.2 + 386.0 / (1+69.9 * np.power(q, 1.08))
    
        T_c_f = 1.0 / (1.0 + np.power(xx / 5.4, 4))
        T_c = T_c_f * T_c_ln_beta / (T_c_ln_beta + T_c_C_noalpha * q*q) + (1 - T_c_f) * T_c_ln_beta / (T_c_ln_beta + T_c_C_alpha * q *q)
        
        s_tilde = self.sound_horizon * np.power(1 + np.power(self.beta_node / xx, 3), -1./3.)
        xx_tilde = k * s_tilde

        T_b_T0 = T_c_ln_nobeta / (T_c_ln_nobeta + T_c_C_noalpha * q * q)
        T_b = np.sin(xx_tilde) / (xx_tilde) * (T_b_T0 / (1 + np.power(xx / 5.2, 2)) + self.alpha_b / (1 + np.power(self.beta_b / xx, 3)) * np.exp(-np.power(k / self.k_silk, 1.4)))
        
        f_baryon = self.obhh / self.omhh
        T_full = f_baryon * T_b + (1 - f_baryon) * T_c;
    
        # Now to store these transfer functions
        return T_full, T_b, T_c

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        