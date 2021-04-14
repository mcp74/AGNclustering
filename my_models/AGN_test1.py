from halotools.empirical_models import PrimGalpropModel
#from ..component_model_templates import PrimGalpropModel
import numpy as np

#class SMBH_mass(object):
class SMBH_mass(PrimGalpropModel):

    def __init__(self,**kwargs):

        #self._mock_generation_calling_sequence = ['assign_mbh']
        #self._galprop_dtypes_to_allocate = np.dtype([('smbh_mass', 'f4')])
        #self.list_of_haloprops_needed = ['halo_mpeak']
        
        super(SMBH_mass, self).__init__(
            galprop_name='smbh_mass', **kwargs)

        self.param_dict.update({'norm': 7.60106435})


    def mean_smbh_mass(self, **kwargs):
        table = kwargs['table']
        halo_mass = kwargs['table'][self.prim_haloprop_key]
        log_halo_mass = np.log10(halo_mass)
        #log_halo_mpeak = np.log10(table['halo_mpeak'])
        #log_stellar_mass = _log_mgal_b19(log_halo_mass)
        #smbh_mass = 10**(_log_mbh_s17(log_stellar_mass))
        smbh_mass = 10**self.mean_smbh_mass_analytic(log_halo_mass, self.param_dict['norm'])
        table['smbh_mass'][:] = smbh_mass
        return smbh_mass

    def mean_smbh_mass_analytic(self, x, norm = 7.60106435):
        delta=0.407
        alpha=4.90392517
        beta=0.69581883 
        M1=12.00794823
        gamma=-0.37902637
        return norm - np.log10( 10**(-alpha * (x - M1)) + 10**(-beta * (x - M1))) + gamma * np.exp(-.5 * ( (x - M1)/delta)**2)

        
class EddingtonRatio(object):

    def __init__(self):
        self._mock_generation_calling_sequence = ['assign_eddington_ratio']
        self._galprop_dtypes_to_allocate = np.dtype([('smbh_eddington_ratio', 'f4')])
    
        self.param_dict = ({'edd_star': 10**-1.84,
            'xi_star': 10**(-1.65),
            'delta_1': 0.47,
            'delta_2': 2.53,
            'lower_edd_limit': -4})

    def ERDF(self, Edd, **kwargs):
        edd_star = self.param_dict['edd_star']
        xi_star = self.param_dict['xi_star']
        delta_1 = self.param_dict['delta_1']
        delta_2 = self.param_dict['delta_2']
        return xi_star * ((Edd / edd_star)**delta_1 + (Edd / edd_star)**delta_2)**(-1)

    def assign_eddington_ratio(self, **kwargs):
        table = kwargs['table']
        lower_lim = self.param_dict['lower_edd_limit']
        log_Edd_grid = np.linspace(lower_lim, 1, 1000)
        kde_erdf=self.ERDF(10**log_Edd_grid)
        cdf = np.cumsum(kde_erdf)
        cdf = cdf / cdf[-1]
        values = np.random.rand(len(table))
        value_bins = np.searchsorted(cdf, values)
        log_edd_arr = log_Edd_grid[value_bins]
        Eddington_ratio = 10**(log_edd_arr)
        table['smbh_eddington_ratio'][:] = Eddington_ratio


class Luminosity(object):

    def __init__(self):
        self._mock_generation_calling_sequence = ['assign_luminosity']
        self._galprop_dtypes_to_allocate = np.dtype([('smbh_log_luminosity', 'f4')])

    def assign_luminosity(self, **kwargs):
        table = kwargs['table']
        log_smbh_mass = np.log10(table['smbh_mass'])
        log_Eddington_limit = np.log10(1.26e38) + log_smbh_mass #erg/s
        log_luminosity = np.log10(table['smbh_eddington_ratio']) + log_Eddington_limit
        table['smbh_log_luminosity'][:] = log_luminosity