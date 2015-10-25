from __future__ import print_function, division, absolute_import

import numpy as np 
import matplotlib.pyplot as plt 
import os
import sys
import subprocess
from astropy.io import fits
from astropy import time
import DataStructures
from ._utils import combine_orders, read_grid_points, ensure_dir
from .analyzer import GSSP_Analyzer
import logging
import glob

home = os.environ['HOME']
GSSP_EXE = '{}/Applications/GSSP/GSSP_single/GSSP_single'.format(home)
GSSP_ABUNDANCE_TABLES = '{}/Applications/GSSPAbundance_Tables/'.format(home)
GSSP_MODELS = '/media/ExtraSpace/GSSP_Libraries/LLmodels/'

class GSSP_Fitter(object):
    teff_minstep = 100
    logg_minstep = 0.1
    feh_minstep = 0.1
    vsini_minstep = 10
    vmicro_minstep = 0.1
    def __init__(self, filename, gssp_exe=None, abund_tab=None, models_dir=None):
        """
        A python wrapper to the GSSP code (must already be installed)

        Parameters:
        ===========
        filename:       string
                        The filename of the (flattened) fits spectrum to fit.

        gssp_exe:       string (optional)
                        The full path to the gssp executable file

        abund_tab:      string (optional)
                        The full path to the directory containing 
                        GSSP abundance tables.

        models_dir:     string:
                        The full path to the directory containing
                        GSSP atmosphere models.

        Methods:
        ==========
        fit:            Fit the parameters
        
        """

        if gssp_exe is None:
            gssp_exe = GSSP_EXE
        if abund_tab is None:
            abund_tab = GSSP_ABUNDANCE_TABLES
        if models_dir is None:
            models_dir = GSSP_MODELS

        # Read in the file and combine the orders
        orders = self._read_fits_file(filename)
        combined = combine_orders(orders)

        #TODO: Cross-correlate the data to get it close. GSSP might have trouble with huge RVs...

        # Get the object name/date
        header = fits.getheader(filename)
        star = header['OBJECT']
        date = header['DATE-OBS']
        try:
            jd = time.Time(date, format='isot', scale='utc').jd
        except TypeError:
            jd = time.Time('{}T{}'.format(date, header['UT']), format='isot', 
                           scale='utc').jd

        # Save the data to an ascii file
        output_basename = '{}-{}'.format(star.replace(' ', ''), jd)
        np.savetxt('data_sets/{}.txt'.format(output_basename), 
                   np.transpose((combined.x, combined.y)),
                   fmt='%.10f')

        # Save some instance variables
        self.data = combined 
        self.jd = jd
        self.starname = star
        self.output_basename = output_basename
        self.gssp_exe = os.path.abspath(gssp_exe)
        self.abundance_table = abund_tab
        self.model_dir = models_dir
        self.gssp_gridpoints = read_grid_points(models_dir)


    def _run_gssp(self, teff_lims=(7000, 30000), teff_step=1000, 
                  logg_lims=(3.0, 4.5), logg_step=0.5, 
                  feh_lims=(-0.5, 0.5), feh_step=0.5,
                  vsini_lims=(50, 350), vsini_step=50,
                  vmicro_lims=(1, 5), vmicro_step=1, 
                  R=80000, ncores=1):
        """
        Coarsely fit the parameters Teff, log(g), and [Fe/H].
        """
        # First, make sure the inputs are reasonable.
        teff_step = max(teff_step, self.teff_minstep)
        logg_step = max(logg_step, self.logg_minstep)
        feh_step = max(feh_step, self.feh_minstep)
        vsini_step = max(vsini_step, self.vsini_minstep)
        vmicro_step = max(vmicro_step, self.vmicro_minstep)
        teff_lims = (min(teff_lims), max(teff_lims))
        logg_lims = (min(logg_lims), max(logg_lims))
        feh_lims = (min(feh_lims), max(feh_lims))
        vsini_lims = (min(vsini_lims), max(vsini_lims))
        vmicro_lims = (min(vmicro_lims), max(vmicro_lims))
        teff_lims, logg_lims, feh_lims = self._check_grid_limits(teff_lims,
                                                                 logg_lims,
                                                                 feh_lims)

        # Make the input file for GSSP
        inp_file=self._make_input_file(teff_lims=teff_lims, teff_step=teff_step, 
                              logg_lims=logg_lims, logg_step=logg_step,
                              feh_lims=feh_lims, feh_step=feh_step, 
                              vsini_lims=vsini_lims, vsini_step=vsini_step, 
                              vmicro_lims=vmicro_lims, vmicro_step=vmicro_step, 
                              resolution=R)

        # Run GSSP
        subprocess.check_call(['mpirun', '-n', '{}'.format(ncores), 
                               '{}'.format(self.gssp_exe), 
                               '{}'.format(inp_file)])

        # Move the output directory to a new name that won't be overridden
        output_dir = '{}_output'.format(self.output_basename)
        ensure_dir(output_dir)
        for f in glob.glob('output_files/*'):
            subprocess.check_call(['mv', f, '{}/'.format(output_dir)])
        return


    def fit(self, teff_lims=(7000, 30000), teff_step=1000, 
            logg_lims=(3.0, 4.5), logg_step=0.5, 
            feh_lims=(-0.5, 0.5), feh_step=0.5,
            vsini_lims=(50, 350), vsini_step=50,
            vmicro_lims=(1, 5), vmicro_step=1, 
            R=80000, ncores=1, refine=True):
        """ 
        Fit the stellar parameters with GSSP 

        Parameters:
        =============
        par_lims:     iterable with (at least) two objects
                      The limits on the given parameter. 'par' can be one of:

                        1. teff:   The effective temperature
                        2. logg:   The surface gravity
                        3. feh:    The metallicity [Fe/H]
                        4. vsini:  The rotational velocity
                        5. vmicro: The microturbulent velocity

                      The default values are a very large, very course grid.
                      Consider refining based on spectral type first!

        par_step:     float
                      The initial step size to take in the given parameter.
                      'par' can be from the same list as above.

        R:            float
                      The spectrograph resolving power (lambda/delta-lambda)

        ncores:       integer, default=1
                      The number of cores to use in the GSSP run.

        refine:       boolean
                      Should we run GSSP again with a smaller grid after the 
                      initial fit? If yes, the best answers will probably be 
                      better.

        Returns:
        =========
        A pd.Series object with the best parameters 
        """
        # Run GSSP
        self._run_gssp(teff_lims=teff_lims, teff_step=teff_step, 
                       logg_lims=logg_lims, logg_step=logg_step,
                       feh_lims=feh_lims, feh_step=feh_step, 
                       vsini_lims=vsini_lims, vsini_step=vsini_step, 
                       vmicro_lims=vmicro_lims, vmicro_step=vmicro_step, 
                       R=R, ncores=ncores)

        # Look at the output and save the figures
        output_dir = '{}_output'.format(self.output_basename)
        best_pars, figs = GSSP_Analyzer(output_dir).estimate_best_parameters()
        for par in figs.keys():
            fig = figs[par]
            fig.savefig(os.path.join(output_dir, '{}_course.pdf'.format(par)))
        plt.close('all')

        if not refine:
            return best_pars 

        # If we get here, we should restrict the grid near the 
        # best solution and fit again
        teff_lims = self._get_refined_limits(lower=best_pars['1sig_CI_lower_Teff'],
                                             upper=best_pars['1sig_CI_upper_Teff'],
                                             values=self.gssp_gridpoints.teff)
        logg_lims = self._get_refined_limits(lower=best_pars['1sig_CI_lower_logg'],
                                             upper=best_pars['1sig_CI_upper_logg'],
                                             values=self.gssp_gridpoints.logg)
        feh_lims = self._get_refined_limits(lower=best_pars['1sig_CI_lower_feh'],
                                            upper=best_pars['1sig_CI_upper_feh'],
                                            values=self.gssp_gridpoints.feh)
        vsini_lower = best_pars.best_vsini*(1-1.5) + 1.5*best_pars['1sig_CI_lower_vsini']
        vsini_upper = best_pars.best_vsini*(1-1.5) + 1.5*best_pars['1sig_CI_upper_vsini']
        vsini_lims = (max(10, vsini_lower), min(400, vsini_upper))
        vsini_step = max(self.vsini_minstep, (vsini_lims[1] - vsini_lims[0])/10)
        vmicro_lims = (best_pars.micro_turb, best_pars.micro_turb)

        # Rename the files in the output directory so they don't get overwritten
        file_list = ['CCF.dat', 'Chi2_table.dat', 
                     'Observed_spectrum.dat', 'Synthetic_best_fit.rgs']
        ensure_dir(os.path.join(output_dir, 'course_output', ''))
        for f in file_list:
            original_fname = os.path.join(output_dir, f)
            new_fname = os.path.join(output_dir, 'course_output', f)
            subprocess.check_call(['mv', original_fname, new_fname])

        # Run GSSP on the refined grid
        self._run_gssp(teff_lims=teff_lims, teff_step=self.teff_minstep, 
                       logg_lims=logg_lims, logg_step=self.logg_minstep,
                       feh_lims=feh_lims, feh_step=self.feh_minstep, 
                       vsini_lims=vsini_lims, vsini_step=round(vsini_step), 
                       vmicro_lims=vmicro_lims, vmicro_step=vmicro_step, 
                       R=R, ncores=ncores)

        best_pars, figs = GSSP_Analyzer(output_dir).estimate_best_parameters()
        for par in figs.keys():
            fig = figs[par]
            fig.savefig(os.path.join(output_dir, '{}_fine.pdf'.format(par)))
            fig.close()

        return best_pars

    
    def _check_grid_limits_old(self, teff_lims, logg_lims, feh_lims):
        df = self.gssp_gridpoints[['teff', 'logg', 'feh']].drop_duplicates()

        # First, check if the limits are do-able
        lower = df.loc[(df.teff <= teff_lims[0]) & 
                       (df.logg <= logg_lims[0]) &
                       (df.feh <= feh_lims[0])]
        upper = df.loc[(df.teff >= teff_lims[1]) & 
                       (df.logg >= logg_lims[1]) &
                       (df.feh >= feh_lims[1])]
        if len(upper) >= 1 and len(lower) >= 1:
            return teff_lims, logg_lims, feh_lims
        
        # If we get here, there is a problem...
        # Check temperature first:
        if not (len(df.loc[df.teff <= teff_lims[0]]) >= 1 and 
                len(df.loc[df.teff >= teff_lims[1]]) >= 1):
            # Temperature grid is no good.
            low_teff, high_teff = df.teff.min(), df.teff.max()
            print('The temperature grid is not available in the model library!')
            print('You wanted temperatures from {} - {}'.format(*teff_lims))
            print('The model grid extends from {} - {}'.format(low_teff, high_teff))
            new_teff_lims = (max(low_teff, teff_lims[0]),
                             min(high_teff, teff_lims[1]))
            print('Resetting temperature limits to {} - {}'.format(*new_teff_lims))
            return self._check_grid_limits(new_teff_lims, logg_lims, feh_lims)

        # Check log(g) next:
        teff_df = df.loc[(df.teff >= teff_lims[0]) & (df.teff <= teff_lims[1])]
        if not (len(teff_df.loc[df.logg <= logg_lims[0]]) >= 1 and 
                len(teff_df.loc[df.logg >= logg_lims[1]]) >= 1):
            # Temperature grid is no good.
            low_logg, high_logg = df.logg.min(), df.logg.max()
            print('The log(g) grid is not available in the model library!')
            print('You wanted log(g) from {} - {}'.format(*logg_lims))
            print('The model grid extends from {} - {}'.format(low_logg, high_logg))
            new_logg_lims = (max(low_logg, logg_lims[0]),
                             min(high_logg, logg_lims[1]))
            print('Resetting log(g) limits to {} - {}'.format(*new_logg_lims))
            return self._check_grid_limits(teff_lims, new_logg_lims, feh_lims)

        # Finally, check [Fe/H]:
        subset_df = df.loc[(df.teff >= teff_lims[0]) & 
                           (df.teff <= teff_lims[1]) *
                           (df.logg >= logg_lims[0]) &
                           (df.logg <= logg_lims[1])]
        if not (len(subset_df.loc[df.feh <= feh_lims[0]]) >= 1 and 
                len(subset_df.loc[df.feh >= feh_lims[1]]) >= 1):
            # Temperature grid is no good.
            low_feh, high_feh = df.feh.min(), df.feh.max()
            print('The [Fe/H] grid is not available in the model library!')
            print('You wanted [Fe/H] from {} - {}'.format(*feh_lims))
            print('The model grid extends from {} - {}'.format(low_feh, high_feh))
            new_feh_lims = (max(low_feh, feh_lims[0]),
                            min(high_feh, feh_lims[1]))
            print('Resetting [Fe/H] limits to {} - {}'.format(*new_feh_lims))
            return self._check_grid_limits(teff_lims, logg_lims, new_feh_lims)

        # We should never get here
        raise ValueError('Something weird happened while checking limits!')


    def _check_grid_limits(self, teff_lims, logg_lims, feh_lims):
        df = self.gssp_gridpoints[['teff', 'logg', 'feh']].drop_duplicates()

        # First, check if the limits are do-able as is
        lower = df.loc[(df.teff == teff_lims[0]) & (df.feh == feh_lims[0])]
        upper = df.loc[(df.teff == teff_lims[1]) & (df.feh == feh_lims[1])]
        if (lower.logg.min() <= logg_lims[0] and 
            lower.logg.max() >= logg_lims[1] and
            upper.logg.min() <= logg_lims[0] and
            upper.logg.max() >= logg_lims[1]):
            return teff_lims, logg_lims, feh_lims

        # If we get here, there is a problem...
        # Check temperature first:
        low_teff, high_teff = df.teff.min(), df.teff.max()
        if low_teff > teff_lims[0] or high_teff < teff_lims[1]:
            print('The temperature grid is not available in the model library!')
            print('You wanted temperatures from {} - {}'.format(*teff_lims))
            print('The model grid extends from {} - {}'.format(low_teff, high_teff))
            new_teff_lims = (max(low_teff, teff_lims[0]),
                             min(high_teff, teff_lims[1]))
            print('Resetting temperature limits to {} - {}'.format(*new_teff_lims))
            return self._check_grid_limits(new_teff_lims, logg_lims, feh_lims)

        # Check [Fe/H] next
        subset_df = df.loc[(df.teff >= teff_lims[0]) & 
                           (df.teff <= teff_lims[1])]
        low_feh, high_feh = subset_df.feh.min(), subset_df.feh.max()
        if low_feh > feh_lims[0] or high_feh < feh_lims[1]:
            print('The [Fe/H] grid is not available in the model library!')
            print('You wanted [Fe/H] from {} - {}'.format(*feh_lims))
            print('The model grid extends from {} - {}'.format(low_feh, high_feh))
            new_feh_lims = (max(low_feh, feh_lims[0]),
                            min(high_feh, feh_lims[1]))
            print('Resetting [Fe/H] limits to {} - {}'.format(*new_feh_lims))
            return self._check_grid_limits(teff_lims, logg_lims, new_feh_lims)

        # Finally, check log(g)
        subset_df = subset_df.loc[(subset_df.feh >= feh_lims[0]) & 
                                  (subset_df.feh <= feh_lims[1])]
        low_logg, high_logg = subset_df.logg.min(), subset_df.logg.max()
        if low_logg > logg_lims[0] or high_logg < logg_lims[1]:
            print('The log(g) grid is not available in the model library!')
            print('You wanted log(g) from {} - {}'.format(*logg_lims))
            print('The model grid extends from {} - {}'.format(low_logg, high_logg))
            new_logg_lims = (max(low_logg, logg_lims[0]),
                             min(high_logg, logg_lims[1]))
            print('Resetting log(g) limits to {} - {}'.format(*new_logg_lims))
            return self._check_grid_limits(teff_lims, new_logg_lims, feh_lims)

        # We should never get here
        raise ValueError('Something weird happened while checking limits!')




    def _get_refined_limits(self, lower, upper, values):
        """ 
        Get the items in the 'values' array that are just
        less than lower and just more than upper.
        """
        unique_values = sorted(np.unique(values))
        l_idx = np.searchsorted(unique_values, lower, side='left')
        r_idx = np.searchsorted(unique_values, upper, side='right')

        if l_idx > 0:
            l_idx -= 1
        if r_idx < len(unique_values) - 1:
            r_idx += 1
        return unique_values[l_idx], unique_values[r_idx]


    def _read_fits_file(self, fname):
        orders = []
        hdulist = fits.open(fname)
        for i, hdu in enumerate(hdulist[1:]):
            xypt = DataStructures.xypoint(x=hdu.data['wavelength'], 
                                          y=hdu.data['flux'], 
                                          cont=hdu.data['continuum'], 
                                          err=hdu.data['error'])
            xypt.x *= 10 #Convert from nanometers to angstrom
            orders.append(xypt)
        return orders


    def _make_input_file(self, teff_lims, teff_step, logg_lims, logg_step,
                         feh_lims, feh_step, vsini_lims, vsini_step, 
                         vmicro_lims, vmicro_step, resolution):
        """ Make the input file for the given star
        """
        output_string = '{:.1f} {:.0f} {:.1f}\n'.format(teff_lims[0], 
                                                        teff_step, 
                                                        teff_lims[-1])

        output_string += '{:.1f} {:.1f} {:.1f}\n'.format(logg_lims[0],
                                                         logg_step,
                                                         logg_lims[1])

        output_string += '{:.1f} {:.1f} {:.1f}\n'.format(vmicro_lims[0],
                                                         vmicro_step,
                                                         vmicro_lims[1])

        output_string += '{:.1f} {:.1f} {:.1f}\n'.format(vsini_lims[0],
                                                         vsini_step,
                                                         vsini_lims[1])

        output_string += "skip 0.03 0.02 0.07  !dilution factor\n"

        output_string += 'skip {:.1f} {:.1f} {:.1f}\n'.format(feh_lims[0],
                                                              feh_step,
                                                              feh_lims[1])
        
        output_string += 'He 0.04 0.005 0.06   ! Individual abundance\n'

        output_string += '0.0 {:.0f}\n'.format(resolution)

        output_string += '{}\n{}\n'.format(self.abundance_table, self.model_dir)

        output_string += '2 1   !atmosphere model vmicro and mass\n'
        output_string += 'ST    ! model atmosphere chemical composition flag\n'

        dx = self.data.x[1] - self.data.x[0]
        output_string += '1 {:.5f} fit\n'.format(dx)

        output_string += 'data_sets/{}.txt\n'.format(self.output_basename)

        output_string += '0.5 0.99 0.0 adjust ! RV determination stuff\n'

        xmin, xmax = self.data.x[0]-1, self.data.x[-1]+1
        output_string += '{:.1f} {:.1f}\n'.format(xmin, xmax)

        outfilename = '{}.inp'.format(self.output_basename)
        with open(outfilename, 'w') as outfile:
            outfile.write(output_string)

        return outfilename










