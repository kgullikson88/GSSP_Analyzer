from __future__ import print_function, division, absolute_import

import numpy as np 
import DataStructures
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import os
import glob
import logging
import pandas as pd

def combine_orders(xypts, snr=None, xspacing=None, numpoints=None, interp_order=3):
    """
    Function to combine a list of xypoints into a single
      xypoint. Useful for combining several orders/chips
      or for coadding spectra

      ***Optional keywords***
      snr: the spectra will be weighted by the signal-to-noise ratio
           before adding
      xspacing: the x-spacing in the final array
      numpoints: the number of points in the final array. If neither
                 numpoints nor xspacing is given, the x-spacing in the
                 final array will be determined by averaging the spacing
                 in each of the xypoints.
      interp_order: the interpolation order. Default is cubic
    """

    if snr is None or type(snr) != list:
        snr = [1.0] * len(xypts)

    # Find the maximum range of the x data:
    first = np.min([o.x[0] for o in xypts])
    last = np.max([o.x[-1] for o in xypts])
    avg_spacing = np.mean([(o.x[-1] - o.x[0]) / float(o.size() - 1) for o in xypts])

    if xspacing is None and numpoints is None:
        xspacing = avg_spacing
    if numpoints is None:
        if xspacing is None:
            xspacing = avg_spacing
        numpoints = (last - first) / xspacing
    x = np.linspace(first, last, numpoints)

    full_array = DataStructures.xypoint(x=x, y=np.zeros(x.size), err=np.zeros(x.size))
    numvals = np.zeros(x.size, dtype=np.float)  # The number of arrays each x point is in
    normalization = 0.0
    for xypt in xypts:
        #interpolator = ErrorPropagationSpline(xypt.x, xypt.y / xypt.cont, xypt.err / xypt.cont, k=interp_order)
        interpolator = spline(xypt.x, xypt.y/xypt.cont, k=interp_order)
        err_interpolator = spline(xypt.x, xypt.err/xypt.cont, k=interp_order)
        left = np.searchsorted(full_array.x, xypt.x[0])
        right = np.searchsorted(full_array.x, xypt.x[-1], side='right')
        if right < xypt.size():
            right += 1
        numvals[left:right] += 1.0
        val, err = interpolator(full_array.x[left:right]), err_interpolator(full_array.x[left:right])
        full_array.y[left:right] += val
        full_array.err[left:right] += err ** 2

    full_array.err = np.sqrt(full_array.err)
    full_array.y[numvals > 0] /= numvals[numvals > 0]
    return full_array


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

def read_grid_points(basename):
    """ Find what grid points are available. Assumes a very strict naming scheme!
    """
    # First level: metallicities and microturbulent velocities
    top_dirs = [os.path.basename(d) for d in os.listdir(basename) if d.startswith('l')]
    teff_list = []
    logg_list = []
    feh_list = []
    vmicro_list = []
    model_list = []
    for directory in top_dirs:
        sign = -1 if directory[1] == 'm' else 1
        feh = sign * float(directory[2:4])/10
        vmicro = float(directory.split('k')[-1])

        # 2nd level: Individual models
        for model in glob.glob(os.path.join(basename, directory, '*.mod')):
            sections = os.path.basename(model).split('_')
            teff = float(sections[1])
            logg = float(sections[2])/100
            teff_list.append(teff)
            logg_list.append(logg)
            feh_list.append(feh)
            vmicro_list.append(vmicro)
            model_list.append(os.path.join(basename, directory, model))

    return pd.DataFrame(data=dict(teff=teff_list,
                                  logg=logg_list,
                                  feh=feh_list,
                                  vmicro=vmicro_list,
                                  filename=model_list))




