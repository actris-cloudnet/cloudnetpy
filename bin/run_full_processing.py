"""
Example script demonstrating full Cloudnet processing for one day.

Measurement files used in this script can be downloaded from:

https://drive.google.com/open?id=1iJVvgDrQo8JvGSOcxroIZc6x8fSUs4EU


"""
import importlib
from cloudnetpy import categorize
from cloudnetpy.instruments import mira, ceilo
from cloudnetpy import plotting as plot


FILE_PATH = '/home/tukiains/Documents/PYTHON/cloudnetpy/test_data/mace/'


def _get_meta(name):
    # This could come via http API, or user could set by hand if needed.
    return {'name': 'Mace-Head',
            'altitude': 5,
            'institute': 'Finnish Meteorological Institute'}


def main(site):
    """ Main function. """

    def _process_raw(input_id, output_id, fun):
        input_file = _input_file_name(input_id)
        output_file = _output_file_name(output_id)
        fun(input_file, output_file, site_meta)
        return output_file

    def _input_file_name(file_id):
        return f"{FILE_PATH}{date}_{site}_{file_id}.nc"

    def _output_file_name(file_id):
        return f"{FILE_PATH}{file_id}_file.nc"

    date = '20181204'

    site_meta = _get_meta(site)

    # Raw processing
    radar_file = _process_raw('mira_raw', 'radar', mira.mira2nc)
    ceilo_file =_process_raw('chm15k', 'ceilo', ceilo.ceilo2nc)

    # Categorize file
    categorize_input_files = {
        'radar': radar_file,
        'lidar': ceilo_file,
        'model': _input_file_name('ecmwf'),
        'mwr': _input_file_name('hatpro')
        }
    categorize_file = _output_file_name('categorize')
    categorize.generate_categorize(categorize_input_files, categorize_file)

    # Products
    for product in ('classification', 'iwc', 'lwc', 'drizzle'):
        product_file = _output_file_name(product)
        module = importlib.import_module(f"cloudnetpy.products.{product}")
        getattr(module, f"generate_{product}")(categorize_file, product_file)

    # Figures
    plot.generate_figure(categorize_file,
                         ['Z', 'Z_error', 'ldr', 'v', 'width'])
    plot.generate_figure(_output_file_name('classification'),
                         ['target_classification', 'detection_status'])
    plot.generate_figure(_output_file_name('iwc'),
                         ['iwc', 'iwc_error', 'iwc_retrieval_status'])
    plot.generate_figure(_output_file_name('lwc'),
                         ['lwc', 'lwc_error', 'lwc_retrieval_status'],
                         max_y=6)
    plot.generate_figure(_output_file_name('drizzle'),
                         ['Do', 'mu', 'S'],
                         max_y=3)


if __name__ == "__main__":
    main('mace-head')
