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
    return {'name': name.capitalize(),
            'altitude': 5,
            'institute': 'Finnish Meteorological Institute'}


def main(site, plot_figures=True):
    """ Main function. """

    def _process_raw(input_id, output_id, fun):
        print(f"Processing raw {output_id} file...")
        input_file = _input_file_name(input_id)
        output_file = _output_file_name(output_id)
        fun(input_file, output_file, site_meta)
        return output_file

    def _process_product(product_name):
        print(f"Processing {product_name} file...")
        output_file = _output_file_name(product_name)
        module = importlib.import_module(f"cloudnetpy.products.{product_name}")
        getattr(module, f"generate_{product_name}")(categorize_file, output_file)

    def _input_file_name(file_id):
        return f"{FILE_PATH}{date}_{site}_{file_id}.nc"

    def _output_file_name(file_id):
        return f"{FILE_PATH}{file_id}_file.nc"

    def _plot_figures():
        print('Plotting...')
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

    date = '20181204'
    site_meta = _get_meta(site)

    radar_file = _process_raw('mira_raw', 'radar', mira.mira2nc)
    ceilo_file = _process_raw('chm15k', 'ceilo', ceilo.ceilo2nc)

    categorize_input_files = {
        'radar': radar_file,
        'lidar': ceilo_file,
        'model': _input_file_name('ecmwf'),
        'mwr': _input_file_name('hatpro')
        }
    categorize_file = _output_file_name('categorize')
    print(f"Processing categorize file...")
    categorize.generate_categorize(categorize_input_files, categorize_file)

    for product in ('classification', 'iwc', 'lwc', 'drizzle'):
        _process_product(product)

    if plot_figures:
        _plot_figures()

    print('All done. Bye!')


if __name__ == "__main__":
    main('mace-head', plot_figures=True)
