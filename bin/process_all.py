"""Testing script for Cloudnet processing.

Creates similar directory structure and file names as the
Matlab processing environment.

"""
import os
import fnmatch
import gzip
import shutil
import importlib
import configparser
import datetime
from collections import namedtuple
from cloudnetpy import categorize as cat
from cloudnetpy.instruments import mira, rpg
from cloudnetpy.instruments import ceilo
from cloudnetpy import plotting
from cloudnetpy import utils

config = configparser.ConfigParser()
config.read('config.ini')

SITE_ROOT = f"{config['PATH']['root']}{config['SITE']['dir_name']}/"


def main():
    """ Main Cloudnet processing function."""

    _read_site_altitude()

    start_date = _period_to_date('PERIOD_START')
    end_date = _period_to_date('PERIOD_END')

    for date in utils.date_range(start_date, end_date):
        dvec = date.strftime("%Y%m%d")
        print('Date: ', dvec)
        for processing_type in ('radar', 'lidar', 'categorize'):
            _run_processing(processing_type, dvec)
        for product in ('classification', 'iwc-Z-T-method',
                        'lwc-scaled-adiabatic', 'drizzle'):
            try:
                _process_product(product, dvec)
            except RuntimeError as error:
                print(error)
        print(' ')


def _period_to_date(section_name):
    date = [config.getint(section_name, key) for key in ('year', 'month', 'day')]
    return datetime.date(*date)


def _run_processing(process_type, dvec):
    module = importlib.import_module(__name__)
    try:
        getattr(module, f"_process_{process_type}")(dvec)
    except RuntimeError as error:
        print(error)


def _process_radar(dvec):
    def _find_uncalibrated_mira_file():
        input_path = _find_uncalibrated_path(instrument, dvec)
        try:
            file = _find_file(input_path, f"*{dvec}*nc")
        except FileNotFoundError:
            try:
                file = _find_file(input_path, f"*{dvec}*.gz")
            except FileNotFoundError as error:
                raise error
            file = gz_to_nc(file)
        return file

    instrument = 'radar'
    output_file = _build_calibrated_file_name(instrument, dvec)
    if config['INSTRUMENTS'][instrument] == 'mira':
        try:
            input_file = _find_uncalibrated_mira_file()
        except FileNotFoundError:
            raise RuntimeError('Abort: Missing uncalibrated mira file.')
        if _is_good_to_process(instrument, output_file):
            print(f"Calibrating mira cloud radar..")
            mira.mira2nc(input_file, output_file, config['SITE'])
    elif config['INSTRUMENTS'][instrument] == 'rpg-fmcw-94':
        rpg_path = _build_uncalibrated_rpg_path(dvec)
        try:
            _ = _find_file(rpg_path, '*.LV1')
        except FileNotFoundError:
            raise RuntimeError('Abort: Missing uncalibrated rpg .LV1 files.')
        if _is_good_to_process(instrument, output_file):
            print(f"Calibrating rpg-fmcw-94 cloud radar..")
            rpg.rpg2nc(rpg_path, output_file, dict(config.items('SITE')))


def _build_uncalibrated_rpg_path(dvec):
    year, month, day = _split_date(dvec)
    rpg_model = config['INSTRUMENTS']['radar']
    return f"{SITE_ROOT}uncalibrated/{rpg_model}/Y{year}/M{month}/D{day}/"


def _process_lidar(dvec):
    instrument = 'lidar'
    input_path = _find_uncalibrated_path(instrument, dvec)
    try:
        input_file = _find_file(input_path, f"*{dvec}*")
    except FileNotFoundError:
        raise RuntimeError('Abort: Missing uncalibrated lidar file.')
    output_file = _build_calibrated_file_name(instrument, dvec)
    if _is_good_to_process(instrument, output_file):
        print(f"Calibrating {config['INSTRUMENTS'][instrument]} lidar..")
        ceilo.ceilo2nc(input_file, output_file, dict(config.items('SITE')))


def _process_categorize(dvec):
    output_file = _build_categorize_file_name(dvec)
    if _is_good_to_process('categorize', output_file):
        try:
            input_files = {
                'radar': _find_calibrated_file('radar', dvec),
                'lidar': _find_calibrated_file('lidar', dvec),
                'mwr': _find_mwr_file(dvec),
                'model': _find_calibrated_file('model', dvec)}
        except FileNotFoundError:
            raise RuntimeError('Input files missing. Cannot process categorize file.')
        try:
            print(f"Processing categorize file..")
            cat.generate_categorize(input_files, output_file)
        except RuntimeError as error:
            raise error
        image_name = _make_image_name(output_file)
        if _is_good_to_plot('categorize', image_name):
            print(f"Generating categorize quicklook..")
            plotting.generate_figure(output_file, ['Z', 'v', 'ldr', 'width', 'beta', 'lwp'],
                                     image_name=image_name, show=False)


def _process_product(product, dvec):
    try:
        categorize_file = _find_categorize_file(dvec)
    except FileNotFoundError:
        raise RuntimeError(f"Failed to process {product}. Categorize file is missing.")
    output_file = _build_product_name(product, dvec)
    product_prefix = product.split('-')[0]
    module = importlib.import_module(f"cloudnetpy.products.{product_prefix}")
    if _is_good_to_process(product, output_file):
        print(f"Processing {product} product..")
        getattr(module, f"generate_{product_prefix}")(categorize_file, output_file)
    image_name = _make_image_name(output_file)
    if _is_good_to_plot(product, image_name):
        print(f"Generating {product} quicklook..")
        fields, max_y = _get_product_fields_in_plot(product_prefix)
        plotting.generate_figure(output_file, fields, image_name=image_name,
                                 show=config.getboolean('MISC', 'show_plot'),
                                 max_y=max_y)


def _get_product_fields_in_plot(product, max_y=12):
    if product == 'classification':
        fields = ['target_classification', 'detection_status']
    elif product == 'iwc':
        fields = ['iwc', 'iwc_error', 'iwc_retrieval_status']
    elif product == 'lwc':
        fields = ['lwc', 'lwc_error', 'lwc_retrieval_status']
        max_y = 8
    elif product == 'drizzle':
        fields = ['Do', 'mu', 'S']
        max_y = 4
    else:
        fields = []
    return fields, max_y


def _build_calibrated_file_name(instrument, dvec):
    output_path = _find_calibrated_path(instrument, dvec)
    return _get_nc_name(output_path, config['INSTRUMENTS'][instrument], dvec)


def _build_categorize_file_name(dvec):
    output_path = _find_categorize_path(dvec)
    return _get_nc_name(output_path, 'categorize', dvec)


def _build_product_name(product, dvec):
    output_path = _find_product_path(product, dvec)
    return _get_nc_name(output_path, product, dvec)


def _is_good_to_process(process_type, output_file):
    is_file = os.path.isfile(output_file)
    process_level = config.getint('PROCESS_LEVEL', process_type)
    process_always = process_level == 2
    process_if_missing = process_level == 1 and not is_file
    return process_always or process_if_missing


def _is_good_to_plot(process_type, image_name):
    is_file = os.path.isfile(image_name)
    quicklook_level = config.getint('QUICKLOOK_LEVEL', process_type)
    plot_always = quicklook_level == 2
    process_if_missing = quicklook_level == 1 and not is_file
    return plot_always or process_if_missing


def _find_mwr_file(dvec):
    _, month, day = _split_date(dvec)
    prefix = _find_uncalibrated_path('mwr', dvec)
    hatpro_path = f"{prefix}{month}/{day}/"
    try:
        return _find_file(hatpro_path, f"*{dvec[2:]}*LWP*")
    except FileNotFoundError:
        if config['INSTRUMENTS']['radar'] == 'rpg-fmcw-94':
            return _find_calibrated_file('radar', dvec)
        raise FileNotFoundError


def _find_uncalibrated_file(instrument, dvec):
    file_path = _find_uncalibrated_path(instrument, dvec)
    return _find_file(file_path, f"*{dvec}*")


def _find_calibrated_file(instrument, dvec):
    file_path = _find_calibrated_path(instrument, dvec)
    return _find_file(file_path, f"*{dvec}*.nc")


def _find_categorize_file(dvec):
    file_path = _find_categorize_path(dvec)
    return _find_file(file_path, f"*{dvec}*.nc")


def _find_product_file(product, dvec):
    file_path = _find_product_path(product, dvec)
    return _find_file(file_path, f"*{dvec}*.nc")


def _find_uncalibrated_path(instrument, dvec):
    year = _get_year(dvec)
    path_all = _get_uncalibrated_paths(config['INSTRUMENTS'])
    path_instrument = getattr(path_all, instrument)
    return f"{path_instrument}{year}/"


def _find_calibrated_path(instrument, dvec):
    year = _get_year(dvec)
    path_all = _get_calibrated_paths(config['INSTRUMENTS'])
    path_instrument = getattr(path_all, instrument)
    output_path = f"{path_instrument}{year}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path


def _find_categorize_path(dvec):
    year = _get_year(dvec)
    categorize_path = f"{SITE_ROOT}/processed/categorize/{year}/"
    if not os.path.exists(categorize_path):
        os.makedirs(categorize_path)
    return categorize_path


def _find_product_path(product, dvec):
    year = _get_year(dvec)
    product_path = f"{SITE_ROOT}/products/{product}/{year}/"
    if not os.path.exists(product_path):
        os.makedirs(product_path)
    return product_path


def gz_to_nc(gz_file):
    """Unzips *.gz file to *.nc file."""
    nc_file = gz_file.replace('gz', 'nc')
    with gzip.open(gz_file, 'rb') as file_in:
        with open(nc_file, 'wb') as file_out:
            shutil.copyfileobj(file_in, file_out)
    return nc_file


def _get_uncalibrated_paths(instruments):
    Paths = namedtuple('Paths', ['radar', 'lidar', 'mwr'])
    prefix = f"{SITE_ROOT}uncalibrated/"
    return Paths(radar=f"{prefix}{instruments['radar']}/",
                 lidar=f"{prefix}{instruments['lidar']}/",
                 mwr=f"{prefix}{instruments['mwr']}/")


def _get_calibrated_paths(instruments):
    Paths = namedtuple('Paths', ['radar', 'lidar', 'model'])
    prefix = f"{SITE_ROOT}calibrated/"
    return Paths(radar=f"{prefix}{instruments['radar']}/",
                 lidar=f"{prefix}{instruments['lidar']}/",
                 model=f"{prefix}{instruments['model']}/")


def _get_nc_name(file_path, prefix, dvec):
    return f"{file_path}{dvec}_{config['SITE']['dir_name']}_{prefix}.nc"


def _make_image_name(output_file):
    return output_file.replace('.nc', '.png')


def _find_file(file_path, wildcard):
    files = os.listdir(file_path)
    for file in files:
        if fnmatch.fnmatch(file, wildcard):
            return file_path + file
    raise FileNotFoundError


def _split_date(dvec):
    year = _get_year(dvec)
    month = _get_month(dvec)
    day = _get_day(dvec)
    return year, month, day


def _get_year(dvec):
    return str(dvec[:4])


def _get_month(dvec):
    return str(dvec[4:6])


def _get_day(dvec):
    return str(dvec[6:8])


def _read_site_altitude():
    site = config['SITE']
    altitude = utils.get_site_information(site['dir_name'], 'altitude')
    site['altitude'] = str(altitude)


if __name__ == "__main__":
    main()
