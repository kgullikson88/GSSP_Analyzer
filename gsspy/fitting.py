from __future__ import print_function, division, absolute_import

import numpy as np 
import matplotlib.pyplot as plt 
import os
import sys
import subprocess
from astropy.io import fits
from astropy import time
import DataStructures
from ._utils import combine_orders

home = os.environ['HOME']
GSSP_EXE = '{}/Applications/GSSP/GSSP_single/GSSP_single'.format(home)
GSSP_ABUNDANCE_TABLES = '{}/Applications/GSSPAbundance_Tables/'.format(home)
GSSP_MODELS = '/media/ExtraSpace/GSSP_Libraries/LLmodels/'

class GSSP_Fitter(object):
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

        self.data = combined 
        self.jd = jd
        self.starname = star
        self.output_basename = output_basename
        self.gssp_exe = gssp_exe
        self.abundance_table = abund_tab
        self.model_dir = models_dir


    def _coarse_fit(self, teff_lims=(7000, 30000), teff_step=1000, 
                    logg_lims=(3.0, 4.5), logg_step=0.5, 
                    feh_lims=(-1.0, 0.5), feh_step=0.5,
                    vsini_lims=(50, 350), vsini_step=50
                    vmicro_lims=(1, 5), vmicro_step=1, 
                    R=None, ncores=1):
        """
        Coarsely fit the parameters Teff, log(g), and [Fe/H].
        """
        # First, make the input file for GSSP
        inp_file=self._make_input_file(teff_lims=teff_lims, teff_step=teff_step, 
                              logg_lims=logg_lims, logg_step=logg_step,
                              feh_lims=feh_lims, feh_step=feh_step, 
                              vsini_lims=vsini_lims, vsini_step=vsini_step, 
                              vmicro_lims=vmicro_lims, vmicro_step=vmicro_step, 
                              resolution=R)

        # Run GSSP
        subprocess.check_call(['mpirun', '-n', '{}'.format(ncores), 
                               './{}'.format(self.gssp_exe), 
                               '{}'.format(inp_file)])

        # Move the output directory to a new name that won't be overridden
        output_dir = '{}_coarse_output'.format(self.output_basename)
        subprocess.check_call(['mv', 'output_files', '{}'.format(output_dir)])





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

        output_string += " skip 0.03 0.02 0.07  !dilution factor\n"

        output_string += 'skip {:.1f} {:.1f} {:.1f}\n'.format(feh_lims[0],
                                                              feh_step,
                                                              feh_lims[1])
        
        output_string += 'He 0.04 0.005 0.06   ! Individual abundance\n')

        output_string += '0.0 {:.0f}\n'.format(resolution)

        output_string += '{}\n{}\n'.format(self.abundance_table, self.model_dir)

        output_string += '2 1   !atmosphere model vmicro and mass\n'
        output_string += 'ST    ! model atmosphere chemical composition flag\n'

        dx = self.data.x[1] - self.data.x[0]
        output_string += '1 {:.5f} fit\n'.format(dx)

        output_string += 'data_sets/{}.txt\n'.format(self.output_basename)

        output_string += '0.5 0.99 0.0 adjust ! RV determination stuff\n'

        output_string += '{.1f} {:.1f}\n'.format(data.x[0]-10, data.x[-1]+10)

        outfilename = '{}.inp'.format(self.output_basename)
        with open(outfilename, 'w') as outfile:
            outfile.write(output_string)

        return outfilename










