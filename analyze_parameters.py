import warnings
warnings.warn('Use gsspy.analyzer.GSSP_Analyzer instead!', DeprecationWarning)

import pandas as pd 
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt 
import os

# Default labels for the Chi^2 output table
CHI2_LABELS = ['feh', 'Teff', 'logg', 'micro_turb', 'vsini', 
               'chi2_inter', 'contin_factor', 'chi2', 'chi2_1sig']

# Which labels are parameters (again, default)
PAR_LABELS = ['feh', 'Teff', 'logg', 'micro_turb', 'vsini', 'dilution']

    
def read_chi2_output(basedir, labels=CHI2_LABELS):
    """Reads the Chi2_Table output, assumed to be located in basedir
    """
    if not basedir.endswith('/'):
        basedir += '/'
    fname = '{}Chi2_table.dat'.format(basedir)

    try:
        df = pd.read_fwf(fname, header=None, names=labels)
    except IOError as e:
        logging.warning('File {} not found!'.format(fname))
        raise e

    return df


def get_best_grid_pars(df, par_labels=PAR_LABELS):
    """
    Just finds the best set of parameters (lowest chi2) within the grid
    The parameters to search are given in par_labels as an iterable
    """
    best_row = df.sort('chi2', ascending=True).ix[0]
    best_pars = {}
    for par in par_labels:
        if par in best_row:
            best_pars[par] = best_row[par]
    # Add the chi^2 information
    best_pars['chi2'] = best_row['chi2']
    best_pars['chi2_1sig'] = best_row['chi2_1sig']
    
    return pd.Series(data=best_pars)


def estimate_best_parameters(grid, best_grid_pars=None):
    """
    Estimate the best parameters by interpolating the grid

    parameters:
    ============
    - grid:  pd.DataFrame
             Contains the full grid. Get this by calling read_chi2_output()

    - best_grid_pars: pd.Series or dict
                      Contains the parameters of the best grid point.
                      If not given, it will be calculated by calling get_best_grid_pars()
                      with the default arguments

    returns:
    =========
    pd.Series object with the best parameter and associated uncertainties
      for each parameter in best_grid_pars
    """
    if best_grid_pars is None:
        best_grid_pars = get_best_grid_pars(grid)

    fig, axes = plt.subplots(3, 2)
    axes = axes.flatten()
    parameters = [p for p in best_grid_pars.index if 'chi2' not in p]
    for i, par in enumerate(parameters):
        ax = axes[i]
        print(par)

        # Get all the other parameters
        other_pars = [p for p in parameters if p != par]

        # Get the chi^2 dependence on the current parameter alone
        cond = np.all([grid[p] == best_grid_pars[p] for p in other_pars], axis=0)
        par_dependence = grid[cond][[par, 'chi2']]
        if len(par_dependence) < 2:
            continue

        # Fit the dependence to a polynomial
        polypars = np.polyfit(par_dependence[par], par_dependence['chi2'] - best_grid_pars['chi2_1sig'], 2)
        chi2_fcn = np.poly1d(polypars)
        roots = sorted(np.roots(polypars))
        minimum = get_minimum(chi2_fcn, search_range=roots)
        if len(minimum) == 1:
            minimum = minimum[0]
        else:
            chi2_vals = chi2_fcn(minimum)
            minimum = minimum[np.argmin(chi2_vals)]

        # Plot
        ax.scatter(par_dependence[par], par_dependence['chi2'], marker='x', color='red')
        ax.scatter(minimum, chi2_fcn(minimum) + best_grid_pars['chi2_1sig'], marker='o', color='blue')
        x = np.linspace(par_dependence[par].min(), par_dependence[par].max(), 25)
        ax.plot(x, chi2_fcn(x) + best_grid_pars['chi2_1sig'], 'g--')
        ax.set_xlabel(par)
        ax.set_ylabel('$\chi^2$')

        # Save the best_parameters
        best_grid_pars['best_{}'.format(par)] = minimum
        best_grid_pars['1sig_CI_lower_{}'.format(par)] = min(roots)
        best_grid_pars['1sig_CI_upper_{}'.format(par)] = max(roots)


    plt.tight_layout()
    plt.show()
    return best_grid_pars


def get_minimum(poly, search_range=None):
    """ Get the minimum values of a polynomial.

    parameters:
    ===========
    - poly:   np.poly1d instance
    - search_range: list of size 2
                    Should contain the minimum and maximum search range in index 0 and 1 (respectively)

    returns:
    ========
    A numpy.ndarray with the minimum values
    """
    # Get the (real-valued) critical points
    crit = poly.deriv().r 
    r_crit = crit[crit.imag == 0].real

    # Remove points outside of search_range, if given
    if search_range is not None:
        r_crit = r_crit[(r_crit > search_range[0]) & (r_crit < search_range[1])]

    test = poly.deriv(2)(r_crit)  # Find the second derivative at each critical point
    return r_crit[test > 0]


def plot_best_model(basename):
    obs_spec = np.loadtxt(os.path.join(basename, 'Observed_spectrum.dat'), unpack=True)
    model_spec = np.loadtxt(os.path.join(basename, 'Synthetic_best_fit.rgs'), usecols=(0,1), unpack=True)
    fig, ax = plt.subplots(1, 1, figsize=(12,7))
    ax.plot(obs_spec[0], obs_spec[1], 'k-', alpha=0.7, label='Observed spectrum')
    ax.plot(model_spec[0], model_spec[1], 'r-', alpha=0.8, label='Model Spectrum')
    ax.set_xlabel('Wavelength ($\AA$)')
    ax.set_ylabel('Normalized Flux')

    leg = ax.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.show()




if __name__ == '__main__':
    chi2 = read_chi2_output(sys.argv[1])
    best_pars = estimate_best_parameters(chi2)
    print(best_pars)
