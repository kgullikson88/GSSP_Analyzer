from __future__ import print_function, division, absolute_import

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import os
import logging
from ._utils import get_minimum

# Default labels for the Chi^2 output table
CHI2_LABELS = ['feh', 'Teff', 'logg', 'micro_turb', 'vsini', 
               'chi2_inter', 'contin_factor', 'chi2', 'chi2_1sig']

# Which labels are parameters (again, default)
PAR_LABELS = ['feh', 'Teff', 'logg', 'micro_turb', 'vsini', 'dilution']

class GSSP_Analyzer(object):
    def __init__(self, basedir, chi2_labels=None, par_labels=None):
        """
        Analyze the output of a GSSP_single run.

        Parameters:
        ===========
        basedir:       string
                       The name of the GSSP output directory. 

        chi2_labels:   iterable, optional
                       Labels to apply to the columns in the 'Chi2_table.dat',
                       which is found in basedir

        par_labels:    iterable, optional
                       The names of the parameters that were fit. This is 
                       mostly the same as chi2_labels, but without the chi^2
                       columns

        """
        if chi2_labels is None:
            chi2_labels = CHI2_LABELS
        if par_labels is None:
            par_labels = PAR_LABELS

        fname = os.path.join(basedir, 'Chi2_table.dat')
        try:
            df = pd.read_fwf(fname, header=None, names=chi2_labels)
        except IOError as e:
            logging.warning('File {} not found!'.format(fname))
            raise e

        self.chi2_labels = chi2_labels
        self.par_labels = par_labels
        self.chi2_df = df
        self.basedir = basedir
        return

    def estimate_best_parameters(self):
        """
        Estimate the best parameters by interpolating the grid

        Returns:
        =========
        pd.Series object with the best parameter and associated uncertainties
          for each parameter

        A tuple of matplotlib.Figure instances with plots for each parameter.
        """
        best_grid_pars = self._get_best_grid_pars()

        parameters = [p for p in self.par_labels if p in self.chi2_df.columns]
        figures = {}
        for i, par in enumerate(parameters):
            logging.debug('Slicing to find best {}'.format(par))

            # Get all the other parameters
            other_pars = [p for p in parameters if p != par]

            # Get the chi^2 dependence on the current parameter alone
            cond = np.all([self.chi2_df[p] == best_grid_pars[p] for p in other_pars], axis=0)
            par_dependence = self.chi2_df[cond][[par, 'chi2']]
            if len(par_dependence) < 2:
                continue
            logging.debug(par_dependence)

            # Fit the dependence to a polynomial
            polypars = np.polyfit(par_dependence[par], 
                                  par_dependence['chi2']-best_grid_pars['chi2_1sig'], 
                                  2)
            chi2_fcn = np.poly1d(polypars)
            roots = sorted(np.roots(polypars))
            minimum = get_minimum(chi2_fcn, search_range=roots)
            if len(minimum) == 1:
                minimum = minimum[0]
            elif len(minimum) > 1:
                chi2_vals = chi2_fcn(minimum)
                minimum = minimum[np.argmin(chi2_vals)]
            else:
                minimum = par_dependence.sort_values(by='chi2')['logg'].values[0]

            # Plot
            fig, ax = plt.subplots(1, 1)
            ax.scatter(par_dependence[par], par_dependence['chi2'], 
                       marker='x', color='red')
            ax.scatter(minimum, chi2_fcn(minimum) + best_grid_pars['chi2_1sig'], 
                       marker='o', color='blue')
            x = np.linspace(par_dependence[par].min(), par_dependence[par].max(), 25)
            ax.plot(x, chi2_fcn(x) + best_grid_pars['chi2_1sig'], 'g--')
            ax.set_xlabel(par)
            ax.set_ylabel('$\chi^2$')

            # Save the best_parameters
            best_grid_pars['best_{}'.format(par)] = minimum
            best_grid_pars['1sig_CI_lower_{}'.format(par)] = min(roots)
            best_grid_pars['1sig_CI_upper_{}'.format(par)] = max(roots)

            figures[par] = fig

        return best_grid_pars, figures   


    def plot_best_model(self):
        """ Plot the observed spectrum with the best model 
        """
        obs_fname = os.path.join(self.basedir, 'Observed_spectrum.dat')
        model_fname = os.path.join(self.basedir, 'Synthetic_best_fit.rgs')
        obs_spec = np.loadtxt(obs_fname, unpack=True)
        model_spec = np.loadtxt(model_fname, usecols=(0,1), unpack=True)

        fig, ax = plt.subplots(1, 1, figsize=(12,7))
        ax.plot(obs_spec[0], obs_spec[1], 'k-', alpha=0.7, label='Observed spectrum')
        ax.plot(model_spec[0], model_spec[1], 'r-', alpha=0.8, label='Model Spectrum')
        ax.set_xlabel('Wavelength ($\AA$)')
        ax.set_ylabel('Normalized Flux')

        leg = ax.legend(loc='best', fancybox=True)
        leg.get_frame().set_alpha(0.5)

        plt.show()



    def _get_best_grid_pars(self):
        """
        Finds the best set of parameters (lowest chi2) within the grid
        The parameters to search are given in self.par_labels as an iterable
        """
        best_row = self.chi2_df.sort('chi2', ascending=True).ix[0]
        best_pars = {}
        for par in self.par_labels:
            if par in best_row:
                best_pars[par] = best_row[par]
        # Add the chi^2 information
        best_pars['chi2'] = best_row['chi2']
        best_pars['chi2_1sig'] = best_row['chi2_1sig']
        
        return pd.Series(data=best_pars)