import os, re, subprocess
from ConfigParser import RawConfigParser
from StringIO import StringIO
from tempfile import mkstemp
import time
from numpy import loadtxt
import glob

#def load(executable='C:\\camb_2\\camb.exe', defaults=None):
def load(executable='/cygdrive/c/camb/camb.exe', defaults=None, debug=False):
    """
    
    Prepare a CAMB executable to be called from Python.
    
    Parameters
    ----------
    executable, optional : path to a CAMB executable. 
                           (default: the built-in CAMB, if it was installed) 

    defaults, optional : string, filename, or dict containing a default ini file
                         to be used for unspecified parameters 
                         (default: camb4py._defaults)
                         
                         
    Returns
    -------
    camb object which can be called with a list of parameters
    
    """
    return camb_disk(executable,defaults, debug)



class camb(object):
    
    def __init__(self, executable=None, defaults=None, debug=False):
        self.defaults = read_ini(defaults or _defaults)
        self.debug = debug
        if executable is None:
            raise Exception(("Couldn't find the default camb executable. Please specify one with 'executable=/path/to/camb'\n"))
        
        self._cygwin = "cygdrive" in executable
        self._fileIndex = 1 if self._cygwin else 0
        
        self.executable = executable
        self.output_names = _output_names
    
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


    def derivative(self, dparam, params, epsilon=None):
        """Get a derivative."""
        params = self._apply_defaults(params)
        try:
            x0 = float(params[dparam])
        except:
            raise Exception("Can't take derivative of non-numerical parameter '%s'=%s"%(dparam,params[dparam]) )
        else:
            params[dparam] = x0 - epsilon/2
            d1 = self(**params)
            params[dparam] = x0 + epsilon/2
            d0 = self(**params)
        
            for k,v in d1.items():
                if k not in ['stdout','misc']: v[:,1:] = (v[:,1:] - d0[k][:,1:])/epsilon
            
            d1['stdout'] = (d0['stdout'],d1['stdout'])
            d1['misc'] = (d0['misc'],d1['misc'])
            return d1
        
    def _apply_defaults(self, params):
        """Get params after applying defaults and removing output files"""
        p = self.defaults.copy()
        p.update(params)
        for k in self.output_names: p.pop(k,None)
        return p
        
    def _getTempFile(self, suffix=None):
        fd, original = mkstemp(suffix=suffix)
        os.close(fd)
        original = self._get_actual_filename(original)
        cygwined = original.replace('\\', '/').replace('C:', '/cygdrive/c')
        return [original, cygwined, fd]

    def _get_tmp_files(self, p):
        output_files = []
        
        if try_str2bool(p['get_scalar_cls']): output_files += ['scalar_output_file']
        if try_str2bool(p['get_vector_cls']): output_files += ['vector_output_file']
        if try_str2bool(p['get_tensor_cls']): output_files += ['tensor_output_file']
        if try_str2bool(p['do_lensing']) and try_str2bool(p['get_scalar_cls']): output_files += ['lensed_output_file', 'lensed_output_file']
        if try_str2bool(p['get_transfer']): output_files += ['transfer_filename(1)', 'transfer_matterpower(1)']
        
        output_files = {k: self._getTempFile(suffix='_%s'%k) for k in output_files}
        param_file = self._getTempFile(suffix='_param')

        return output_files, param_file
    
    def _get_cygwin_command(self, paramfile):
        c = 'C:\\cygwin64\\bin\\bash.exe --login -c "cd %s && %s %s"' % (self.executable[:self.executable.rfind('/')], self.executable, paramfile.replace('\\', '/').replace('C:', '/cygdrive/c'))
        if self.debug: print paramfile
        return c
        
    def _get_windows_command(self, paramfile):
        c = "%s %s" % (self.executable, paramfile)
        if self.debug: print c
        return c
        
    def _call_camb(self, paramfile, output_files, result=None):
        if result is None: result = {}
        try:
            if (self._cygwin):
                result['stdout'] = subprocess.check_output(self._get_cygwin_command(paramfile), stderr=subprocess.STDOUT)
            else:
                result['stdout'] = subprocess.check_output(self._get_windows_command(paramfile), cwd=os.path.dirname(self.executable), stderr=subprocess.STDOUT)            
        except subprocess.CalledProcessError as e:
            print 'Warning: CAMB failed with exit code %s'%e.returncode
            result['stdout'] = e.output
        result['misc'] = self._parse_stdout(result['stdout'])
        result['misc']['outputFiles'] = [value[0] for key, value in output_files.iteritems()]
        result['misc']['inputFile'] = paramfile   
        return result

    def _write_ini(self, p, file):
        file.write('\n'.join(['%s = %s'%(k,try_bool2str(v)) for (k,v) in p.items()]+['END','']))

    def _parse_stdout(self,stdout):
        parsed = {}
        for line in stdout.splitlines():
            matches = list(re.finditer('\s*(.+?)\s*=\s*(.+?)(\s|$)',line))
            for m in matches: parsed[m.group(1)]=m.group(2)
        return parsed
    

class camb_disk(camb):
    """Implementation of CAMB which uses regular files on disk for communication."""

    def __call__(self, **params):
        """
        
        Call CAMB and return the output files as well as stdout. 
        
        Parameters
        ----------
        
        **params : all key value pairs are passed to the CAMB ini
        
        """
        
        start_time = time.time()

        params = self._apply_defaults(params)
        
        output_files, param_file = self._get_tmp_files(params)
        for (key,filename) in output_files.items(): 
            params[key]=filename[self._fileIndex]
        with open(param_file[0],'w') as f: 
            self._write_ini(params, f)
        
        result = self._call_camb(param_file[0], output_files)
        for key,filename in output_files.items():
            rkey = self.output_names.get(key,key)
            try: 
                dtypes = float
                if _output_dtypes.get(rkey):
                    dtypes = _output_dtypes[rkey]
                result[rkey] = loadtxt(filename[0], dtype=dtypes)
            except Exception as e: result[rkey] = e
            try: os.remove(filename[0])
            except: print("could not delete %s" % filename[0])
            
        try: os.remove(param_file[0])
        except: print("could not delete %s" % param_file[0])
        
        end_time = time.time()
        if self.debug: print("Elapsed time was %g seconds" % (end_time - start_time))
        return result


    
def get_valid_params(self, sourcedir):
    """Scour CAMB source files for valid parameters"""
    camb_keys=set()
    for f in os.listdir('.'):
        if f.endswith('90'):
            with open(f) as f:
                for line in f:
                    r = re.search("Ini_Read.*File\(.*?,'(.*)'",line,re.IGNORECASE)
                    if r: camb_keys.add(r.group(1))
                    r = re.search("Ini_Read.*\('(.*)'",line,re.IGNORECASE)
                    if r: camb_keys.add(r.group(1))    
                    
    return camb_keys


def try_bool2str(value):
    if value is True: return 'T'
    elif value is False: return 'F'
    else: return value
    
def try_str2bool(value):
    if isinstance(value,str):
        if value.lower() in ['t','true']: return True
        elif value.lower() in ['f','false']: return False
    return value

    
def read_ini(ini):
    """Load an ini file or string into a dictionary."""
    if isinstance(ini,dict): return ini
    if isinstance(ini,str):
        if os.path.exists(ini): ini = open(ini).read()
        config = RawConfigParser()
        config.optionxform=str
        config.readfp(StringIO('[root]\n'+ini))
        return dict(config.items('root'))
    else:
        raise ValueError('Unexpected type for ini file %s'%type(ini))
        




_output_dtypes = {'scalar': [('l', 'int'), ('c_tt', 'float'), ('c_ee', 'float'), ('c_te', 'float'), ('c_phi', 'float'), ('c_phit', 'float')],
                  'lensed': [('l', 'int'), ('unknown1', 'float'), ('unknown2', 'float'), ('unknown3', 'float'), ('unknown4', 'float')],
                  #'transfer_filename(1)': [()]
                  'transfer_matterpower': [('k/h', 'float'), ('power', 'float')]
                  }

_output_names = {'scalar_output_file':'scalar',
                 'vector_output_file':'vector',
                 'tensor_output_file':'tensor',
                 'total_output_file':None,
                 'lensed_output_file':'lensed', 
                 'lens_potential_output_file':'lens_potential',
                 'lensed_total_output_file':None,
                 'transfer_filename(1)':'transfer',
                 'transfer_matterpower(1)':'transfer_matterpower',
                 'fits_filename':None,
                 'output_root':None}



_defaults="""
#Parameters for CAMB

#output_root is prefixed to output file names
output_root = AAAA

#What to do
get_scalar_cls = T
get_vector_cls = F
get_tensor_cls = F
get_transfer   = T

#if do_lensing then scalar_output_file contains additional columns of l^4 C_l^{pp} and l^3 C_l^{pT}
#where p is the projected potential. Output lensed CMB Culs (without tensors) are in lensed_output_file below.
do_lensing     = T

# 0: linear, 1: non-linear matter power (HALOFIT), 2: non-linear CMB lensing (HALOFIT), 
# 3: both non-linear matter power and CMB lensing (HALOFIT) 
do_nonlinear = 0

#Maximum multipole and k*eta. 
#  Note that C_ls near l_max are inaccurate (about 5%), go to 50 more than you need
#  Lensed power spectra are computed to l_max_scalar-100 
#  To get accurate lensed BB need to have l_max_scalar>2000, k_eta_max_scalar > 10000
#  To get accurate lensing potential you also need k_eta_max_scalar > 10000
#  Otherwise k_eta_max_scalar=2*l_max_scalar usually suffices, or don't set to use default
l_max_scalar      = 2200
#k_eta_max_scalar  = 4000

#  Tensor settings should be less than or equal to the above
l_max_tensor      = 1500
k_eta_max_tensor  = 3000

#Main cosmological parameters, neutrino masses are assumed degenerate
# If use_phyical set physical densities in baryons, CDM and neutrinos + Omega_k
use_physical   = T
ombh2          = 0.0226
omch2          = 0.112
omnuh2         = 0.00064
omk            = 0
hubble         = 70

#effective equation of state parameter for dark energy
w              = -1
#constant comoving sound speed of the dark energy (1=quintessence)
cs2_lam        = 1

#varying w is not supported by default, compile with EQUATIONS=equations_ppf to use crossing PPF w-wa model:
#wa             = 0
##if use_tabulated_w read (a,w) from the following user-supplied file instead of above
#use_tabulated_w = F
#wafile = wa.dat

#if use_physical = F set parameters as here
#omega_baryon   = 0.0462
#omega_cdm      = 0.2538
#omega_lambda   = 0.7
#omega_neutrino = 0

temp_cmb           = 2.7255
helium_fraction    = 0.24

#for share_delta_neff = T, the fractional part of massless_neutrinos gives the change in the effective number 
#(for QED + non-instantaneous decoupling)  i.e. the increase in neutrino temperature,
#so Neff = massless_neutrinos + sum(massive_neutrinos)
#For full neutrino parameter details see http://cosmologist.info/notes/CAMB.pdf
massless_neutrinos = 2.046

#number of distinct mass eigenstates
nu_mass_eigenstates = 1
#array of the integer number of physical neutrinos per eigenstate, e.g. massive_neutrinos = 2 1
massive_neutrinos  = 1
#specify whether all neutrinos should have the same temperature, specified from fractional part of massless_neutrinos
share_delta_neff = T
#nu_mass_fractions specifies how Omeganu_h2 is shared between the eigenstates
#i.e. to indirectly specify the mass of each state; e.g. nu_mass_factions= 0.75 0.25
nu_mass_fractions = 1
#if share_delta_neff = F, specify explicitly the degeneracy for each state (e.g. for sterile with different temperature to active)
#(massless_neutrinos must be set to degeneracy for massless, i.e. massless_neutrinos does then not include Deleta_Neff from massive)
#if share_delta_neff=T then degeneracies is not given and set internally
#e.g. for massive_neutrinos = 2 1, this gives equal temperature to 4 neutrinos: nu_mass_degeneracies = 2.030 1.015, massless_neutrinos = 1.015
nu_mass_degeneracies = 

#Initial power spectrum, amplitude, spectral index and running. Pivot k in Mpc^{-1}.
initial_power_num         = 1
pivot_scalar              = 0.05
pivot_tensor              = 0.05
scalar_amp(1)             = 2.1e-9
scalar_spectral_index(1)  = 0.96
scalar_nrun(1)            = 0
scalar_nrunrun(1)         = 0
tensor_spectral_index(1)  = 0
tensor_nrun(1)            = 0
#Three parameterizations (1,2,3) for tensors, see http://cosmologist.info/notes/CAMB.pdf
tensor_parameterization   = 1
#ratio is that of the initial tens/scal power spectrum amplitudes, depending on parameterization
#for tensor_parameterization == 1, P_T = initial_ratio*scalar_amp*(k/pivot_tensor)^tensor_spectral_index
#for tensor_parameterization == 2, P_T = initial_ratio*P_s(pivot_tensor)*(k/pivot_tensor)^tensor_spectral_index
#Note that for general pivot scales and indices, tensor_parameterization==2 as P_T depending on n_s
initial_ratio(1)          = 1
#tensor_amp is used instead if tensor_parameterization == 3, P_T = tensor_amp *(k/pivot_tensor)^tensor_spectral_index
#tensor_amp(1)            = 4e-10

#note vector modes use the scalar settings above


#Reionization, ignored unless reionization = T, re_redshift measures where x_e=0.5
reionization         = T

re_use_optical_depth = T
re_optical_depth     = 0.09
#If re_use_optical_depth = F then use following, otherwise ignored
re_redshift          = 11
#width of reionization transition. CMBFAST model was similar to re_delta_redshift~0.5.
re_delta_redshift    = 1.5
#re_ionization_frac=-1 sets to become fully ionized using YE to get helium contribution
#Otherwise x_e varies from 0 to re_ionization_frac
re_ionization_frac   = -1


#RECFAST 1.5.x recombination parameters;
RECFAST_fudge = 1.14
RECFAST_fudge_He = 0.86
RECFAST_Heswitch = 6
RECFAST_Hswitch  = T

# CosmoMC parameters - compile with RECOMBINATION=cosmorec and link to CosmoMC to use these
#
# cosmorec_runmode== 0: CosmoMC run with diffusion
#                    1: CosmoMC run without diffusion
#                    2: RECFAST++ run (equivalent of the original RECFAST version)
#                    3: RECFAST++ run with correction function of Calumba & Thomas, 2010
#
# For 'cosmorec_accuracy' and 'cosmorec_fdm' see CosmoMC for explanation
#---------------------------------------------------------------------------------------
#cosmorec_runmode        = 0
#cosmorec_accuracy       = 0
#cosmorec_fdm            = 0

#Initial scalar perturbation mode (adiabatic=1, CDM iso=2, Baryon iso=3, 
# neutrino density iso =4, neutrino velocity iso = 5) 
initial_condition   = 1
#If above is zero, use modes in the following (totally correlated) proportions
#Note: we assume all modes have the same initial power spectrum
initial_vector = -1 0 0 0 0

#For vector modes: 0 for regular (neutrino vorticity mode), 1 for magnetic
vector_mode = 0

#Normalization
COBE_normalize = F
##CMB_outputscale scales the output Culs
#To get MuK^2 set realistic initial amplitude (e.g. scalar_amp(1) = 2.3e-9 above) and
#otherwise for dimensionless transfer functions set scalar_amp(1)=1 and use
#CMB_outputscale = 1
CMB_outputscale = 7.42835025e12 

#Transfer function settings, transfer_kmax=0.5 is enough for sigma_8
#transfer_k_per_logint=0 sets sensible non-even sampling; 
#transfer_k_per_logint=5 samples fixed spacing in log-k
#transfer_interp_matterpower =T produces matter power in regular interpolated grid in log k; 
# use transfer_interp_matterpower =F to output calculated values (e.g. for later interpolation)
transfer_high_precision = F
transfer_kmax           = 40
transfer_k_per_logint   = 0
transfer_num_redshifts  = 1
transfer_interp_matterpower = T
transfer_redshift(1)    = 0
transfer_filename(1)    = transfer_out.dat
#Matter power spectrum output against k/h in units of h^{-3} Mpc^3
transfer_matterpower(1) = matterpower.dat

#which variable to use for defining the matter power spectrum and sigma8
#main choices are 2: CDM, 7: CDM+baryon+neutrino, 8: CDM+baryon, 9: CDM+baryon+neutrino+de perts
transfer_power_var = 7

#Output files not produced if blank. make camb_fits to use the FITS setting.
scalar_output_file = scalCls.dat
vector_output_file = vecCls.dat
tensor_output_file = tensCls.dat
total_output_file  = totCls.dat
lensed_output_file = lensedCls.dat
lensed_total_output_file  =lensedtotCls.dat
lens_potential_output_file = lenspotentialCls.dat
FITS_filename      = scalCls.fits

#Bispectrum parameters if required; primordial is currently only local model (fnl=1)
#lensing is fairly quick, primordial takes several minutes on quad core
do_lensing_bispectrum = F
do_primordial_bispectrum = F

#1 for just temperature, 2 with E
bispectrum_nfields = 1
#set slice non-zero to output slice b_{bispectrum_slice_base_L L L+delta}
bispectrum_slice_base_L = 0
bispectrum_ndelta=3
bispectrum_delta(1)=0
bispectrum_delta(2)=2
bispectrum_delta(3)=4
#bispectrum_do_fisher estimates errors and correlations between bispectra
#note you need to compile with LAPACK and FISHER defined to use get the Fisher info
bispectrum_do_fisher= F
#Noise is in muK^2, e.g. 2e-4 roughly for Planck temperature
bispectrum_fisher_noise=0
bispectrum_fisher_noise_pol=0
bispectrum_fisher_fwhm_arcmin=7
#Filename if you want to write full reduced bispectrum (at sampled values of l_1)
bispectrum_full_output_file=
bispectrum_full_output_sparse=F
#Export alpha_l(r), beta_l(r) for local non-Gaussianity
bispectrum_export_alpha_beta=F

##Optional parameters to control the computation speed,accuracy and feedback

#If feedback_level > 0 print out useful information computed about the model
feedback_level = 1

#write out various derived parameters
derived_parameters = T

# 1: curved correlation function, 2: flat correlation function, 3: inaccurate harmonic method
lensing_method = 1
accurate_BB = F


#massive_nu_approx: 0 - integrate distribution function
#                   1 - switch to series in velocity weight once non-relativistic
massive_nu_approx = 1

#Whether you are bothered about polarization. 
accurate_polarization   = T

#Whether you are bothered about percent accuracy on EE from reionization
accurate_reionization   = T

#whether or not to include neutrinos in the tensor evolution equations
do_tensor_neutrinos     = T

#Whether to turn off small-scale late time radiation hierarchies (save time,v. accurate)
do_late_rad_truncation   = T

#Which version of Halofit approximation to use (default currently Takahashi):
#1. original, 2. Bird et al. update, 3. (1) plus fudge from http://www.roe.ac.uk/~jap/haloes/, 4. Takahashi
halofit_version= 

#Computation parameters
#if number_of_threads=0 assigned automatically
number_of_threads       = 0

#Default scalar accuracy is about 0.3% (except lensed BB) if high_accuracy_default=F
#If high_accuracy_default=T the default target accuracy is 0.1% at L>600 (with boost parameter=1 below)
#Try accuracy_boost=2, l_accuracy_boost=2 if you want to check stability/even higher accuracy
#Note increasing accuracy_boost parameters is very inefficient if you want higher accuracy,
#but high_accuracy_default is efficient 

high_accuracy_default=T

#Increase accuracy_boost to decrease time steps, use more k values,  etc.
#Decrease to speed up at cost of worse accuracy. Suggest 0.8 to 3.
accuracy_boost          = 1

#Larger to keep more terms in the hierarchy evolution. 
l_accuracy_boost        = 1

#Increase to use more C_l values for interpolation.
#Increasing a bit will improve the polarization accuracy at l up to 200 -
#interpolation errors may be up to 3%
#Decrease to speed up non-flat models a bit
l_sample_boost          = 1

"""