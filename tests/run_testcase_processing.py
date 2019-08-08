"""
Example script demonstrating full Cloudnet processing for one day.

Measurement files used in this script can be downloaded from:

https://drive.google.com/open?id=1iJVvgDrQo8JvGSOcxroIZc6x8fSUs4EU


"""
import importlib
from cloudnetpy import categorize
from cloudnetpy.instruments import mira, ceilo
from cloudnetpy import plotting as plot


def process_cloudnetpy_raw_files(site, file_path):

    def _process_raw(input_id, output_id, fun):
        print(f"Processing raw {output_id} file...")
        input_file = _input_file_name(file_path, date, site, input_id)
        output_file = _output_file_name(output_id)
        fun(input_file, output_file, site_meta)
        return output_file

    date = '20181204'
    site_meta = _get_meta(site)

    _process_raw('mira_raw', 'radar', mira.mira2nc)
    _process_raw('chm15k_raw', 'ceilo', ceilo.ceilo2nc)


def process_cloudnetpy_categorize(site, file_path):

    def _process_categorize():
        print(f"Processing categorize file...")
        input_files = _get_categorize_input_files()
        output_file = _output_file_name('categorize')
        categorize.generate_categorize(input_files, output_file)
        return output_file

    def _get_categorize_input_files():
        return {'radar': radar_file,
                'lidar': ceilo_file,
                'model': _input_file_name('ecmwf'),
                'mwr': _input_file_name('hatpro')}

    date = '20181204'
    site_meta = _get_meta(site)

    _process_categorize()


def process_cloudnetpy_products(site):

    def _process_product(product_name):
        print(f"Processing {product_name} file...")
        output_file = _output_file_name(product_name)
        module = importlib.import_module(f"cloudnetpy.products.{product_name}")
        getattr(module, f"generate_{product_name}")(categorize_file, output_file)

    date = '20181204'
    site_meta = _get_meta(site)

    for product in ('classification', 'iwc', 'lwc', 'drizzle'):
        _process_product(product)


def _get_meta(name):
    return {'name': name.capitalize(),
            'altitude': 16,
            'institute': 'Finnish Meteorological Institute'}


def _input_file_name(file_path, date, site, file_id):
    return f"{file_path}{date}_{site}_{file_id}.nc"


def _output_file_name(file_path, file_id):
    return f"{file_path}{file_id}_file.nc"

