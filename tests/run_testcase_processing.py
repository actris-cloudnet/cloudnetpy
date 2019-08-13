"""
Example script demonstrating full Cloudnet processing for one day.

Measurement files used in this script can be downloaded from:

https://drive.google.com/open?id=1iJVvgDrQo8JvGSOcxroIZc6x8fSUs4EU


"""
import sys
import importlib
from cloudnetpy.categorize import categorize
from cloudnetpy.instruments import mira, ceilo


def process_cloudnetpy_raw_files(site, file_path):

    def _process_raw(input_id, output_id, fun):
        sys.stdout.write(f"    Processing raw {output_id} file...")
        input_file = _input_raw_file_name(file_path, date, site, input_id)
        output_file = _output_file_name(file_path, output_id)
        fun(input_file, output_file, site_meta)
        sys.stdout.write("    Done.\n")
        return output_file

    date = '20181204'
    site_meta = _get_meta(site)

    _process_raw('mira_raw', 'radar', mira.mira2nc)
    _process_raw('chm15k_raw', 'ceilo', ceilo.ceilo2nc)


def process_cloudnetpy_categorize(site, file_path):

    def _process_categorize():
        sys.stdout.write(f"    Processing categorize file...")
        input_files = _get_categorize_input_files()
        output_file = _output_file_name(file_path, 'categorize')
        categorize.generate_categorize(input_files, output_file)
        sys.stdout.write("    Done.\n")
        return output_file

    def _get_categorize_input_files():
        return {'radar': radar_file,
                'lidar': ceilo_file,
                'model': _input_raw_file_name(file_path, date, site, 'ecmwf'),
                'mwr': _input_raw_file_name(file_path, date, site, 'hatpro')}

    date = '20181204'
    radar_file = _input_file_name(file_path, 'radar')
    ceilo_file = _input_file_name(file_path, 'ceilo')

    _process_categorize()


def process_cloudnetpy_products(file_path):

    def _process_product(product_name):
        sys.stdout.write(f"    Processing {product_name} file...")
        output_file = _output_file_name(file_path, product_name)
        module = importlib.import_module(f"cloudnetpy.products.{product_name}")
        getattr(module, f"generate_{product_name}")(categorize_file, output_file)
        sys.stdout.write("    Done.\n")

    categorize_file = _input_file_name(file_path, 'categorize')
    for product in ('classification', 'iwc', 'lwc', 'drizzle'):
        _process_product(product)


def _get_meta(name):
    return {'name': name.capitalize(),
            'altitude': 16,
            'institute': 'Finnish Meteorological Institute'}


def _input_raw_file_name(file_path, date, site, file_id):
    return f"{file_path}{date}_{site}_{file_id}.nc"


#def _get_meta_from_file():


def _input_file_name(file_path, file_id):
    return f"{file_path}{file_id}_file.nc"


def _output_file_name(file_path, file_id):
    return f"{file_path}{file_id}_file.nc"


