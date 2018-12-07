""" This module is used to generate
categorize (Level 1) product from pre-processed
radar, lidar and MWR files.
"""

# import sys
import numpy as np
import numpy.ma as ma
from scipy import stats
import utils.ncf as ncf


def generate_categorize(input_files, output_file, aux):
    """ Generate Cloudnet Level 1 categorize file.

    Args:
        input_files (tuple): Tuple of strings containing full paths of
                             4 input files (radar, lidar, mwr, model).
        output_file (str): Full path of output file.
        aux (tuple): Tuple of strings including some metadata
                     of the site (site_name, institute).

    """

    TIME_RESOLUTION = 30  # fixed time resolution for now

    rad, rad_vrs = ncf.load_nc(input_files[0])
    lid, lid_vrs = ncf.load_nc(input_files[1])
    mwr, mwr_vrs = ncf.load_nc(input_files[2])
    mod, mod_vrs = ncf.load_nc(input_files[3])

    try:
        freq = get_radar_freq(rad_vrs)
    except (ValueError, KeyError) as error:
        print(error)

    try:
        time = get_time(TIME_RESOLUTION)
    except ValueError as error:
        print(error)

    height = get_altitude_grid(rad_vrs['altitude'][:],
                               rad_vrs['range'][:])

    site_altitude = get_site_altitude(rad_vrs['altitude'][:],
                                      lid_vrs['altitude'][:])

    # average radar variables in time
    fields = ('Zh', 'v', 'ldr', 'width')
    try:
        radar = fetch_radar(rad_vrs, fields, time)
    except KeyError as error:
        print(error)

    vfold = rad_vrs['NyquistVelocity'][:]


def get_radar_freq(vrs):
    """ Return frequency of radar.

    Args:
        vrs: Pointer to radar variables.

    Returns:
        Frequency or radar.

    Raises:
        KeyError: No frequency in the radar file.
        ValueError: Invalid frequency value.

    """
    possible_fields = ('radar_frequency', 'frequency')  # Several possible
    freq = [vrs[field][:] for field in vrs if field in possible_fields]
    if not freq:
        raise KeyError('Missing frequency in the radar file.')
    else:
        freq = freq[0]  # actual data of the masked data
    assert ma.count(freq) == 1, 'Multiple frequencies. Not a radar file??'
    range_1 = 30 < freq < 40
    range_2 = 90 < freq < 100
    if not (range_1 or range_2):
        raise ValueError('Only 35 and 94 GHz radars supported.')
    return float(freq)


def get_site_altitude(alt_radar, alt_lidar):
    """ Return altitude of the measurement site above mean sea level.

    Site altitude is the altitude of radar or lidar, which one is lower.

    Args:
        alt_radar (float): Altitude of radar above mean sea level [km]
        alt_lidar (float): Altitude of lidar above mean sea level [km]

    Returns:
        Altitude of the measurement site.

    """
    return min(alt_radar, alt_lidar)


def get_altitude_grid(alt_radar, range_radar):
    """ Return altitude grid for Cloudnet products.
    Altitude grid is defined as the radar measurement
    grid from the mean sea level.

    Args:
        alt_radar (float): Altitude of radar above mean sea level [km]
        range_radar (nd.array): Altitude grid of radar measurements
                                above instrument [km]

    Returns:
        (nd.array): Altitude grid

    """
    return range_radar + alt_radar


def get_time(reso):
    """ Computes fraction hour time vector 0-24 with user-given
    resolution (in seconds) where 60 is the maximum allowed value.

    Args:
        reso (float): Time resolution in seconds.

    Returns:
        (nd.array): Time vector between 0 and 24.

    Raises:
        ValueError: Bad resolution as input.

    """
    if reso < 1 or reso > 60:
        raise ValueError('Time resolution should be between 0 and 60 [s]')
    step = reso/7200
    return np.arange(step, 24-step, step*2)


def fetch_radar(vrs, fields, time):
    """ Read and rebin radar 2d fields in time.

    Args:
        vrs: Pointer to radar variables
        fields (tuple): Tuple of strings containing radar
                        fields to be averaged.
        time: New time vector.

    Returns:
        (dict): Rebinned radar fields.

    Raises:
        KeyError: Missing field.

    """
    out = {}
    x = vrs['time'][:]
    for field in fields:
        if field in vrs:
            out[field] = rebin_x_2d(x, vrs[field][:], time)
        else:
            raise KeyError(f"No variable '{field}' in the radar file.")
    return out


def rebin_x_2d(x, data, xnew):
    """ Rebin 2D data in x-direction using mean. Handles masked data.

    Args:
        x: A 1-D array of real values.
        data (nd.array): 2-D input data.
        xnew: The new x vector.

    Returns:
        Rebinned field.

    """
    # new binning vector
    edge1 = round(xnew[0] - (xnew[1]-xnew[0])/2)
    edge2 = round(xnew[-1] + (xnew[-1]-xnew[-2])/2)
    edges = np.linspace(edge1, edge2, len(xnew)+1)
    # prepare input/output data
    datai = np.zeros((len(xnew), data.shape[1]))
    data = ma.masked_invalid(data)
    # loop over y
    for ind, values in enumerate(data.T):
        mask = values.mask
        if len(values[~mask]) > 0:
            datai[:, ind], _, _ = stats.binned_statistic(x[~mask],
                                                         values[~mask],
                                                         statistic='mean',
                                                         bins=edges)
    datai[np.isfinite(datai) == 0] = 0
    return ma.masked_equal(datai, 0)
