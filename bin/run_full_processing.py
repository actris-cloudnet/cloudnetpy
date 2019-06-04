"""
Example script demonstrating full Cloudnet processing for one day.

Measurement files used in this script can be downloaded from:

https://drive.google.com/open?id=1iJVvgDrQo8JvGSOcxroIZc6x8fSUs4EU


"""
from cloudnetpy import categorize
from cloudnetpy.products import iwc, lwc, classification, drizzle
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

    date = '20181204'

    site_meta = _get_meta(site)

    # Radar processing
    # ----------------

    input_file = f"{FILE_PATH}{date}_{site}_mira_raw.nc"
    radar_file = f"{FILE_PATH}radar_file.nc"
    mira.mira2nc(input_file, radar_file, site_meta)

    # Ceilometer processing
    # ---------------------

    input_file = f"{FILE_PATH}{date}_{site}_chm15k.nc"
    ceilo_file = f"{FILE_PATH}ceilo_file.nc"
    ceilo.ceilo2nc(input_file, ceilo_file, site_meta)

    # Categorize file
    # ---------------

    categorize_input_files = {
        'radar': radar_file,
        'lidar': ceilo_file,
        'model': f"{FILE_PATH}{date}_{site}_ecmwf.nc",
        'mwr': f"{FILE_PATH}{date}_{site}_hatpro.nc"
        }
    categorize_file = f"{FILE_PATH}categorize_file.nc"
    categorize.generate_categorize(categorize_input_files, categorize_file)

    # Products
    # --------

    classification_file = f"{FILE_PATH}classification_file.nc"
    classification.generate_class(categorize_file, classification_file)

    iwc_file = f"{FILE_PATH}iwc_file.nc"
    iwc.generate_iwc(categorize_file, iwc_file)

    lwc_file = f"{FILE_PATH}lwc_file.nc"
    lwc.generate_lwc(categorize_file, lwc_file)

    drizzle_file = f"{FILE_PATH}drizzle_file.nc"
    drizzle.generate_drizzle(categorize_file, drizzle_file)

    # Figures
    # -------

    plot.generate_figure(categorize_file, ['Z', 'Z_error', 'ldr', 'v', 'width'])
    plot.generate_figure(classification_file, ['target_classification', 'detection_status'])
    plot.generate_figure(iwc_file, ['iwc', 'iwc_error', 'iwc_retrieval_status'])
    plot.generate_figure(lwc_file, ['lwc', 'lwc_error', 'lwc_retrieval_status'], max_y=6)
    plot.generate_figure(drizzle_file, ['Do', 'mu', 'S'], max_y=3)


if __name__ == "__main__":
    main('mace-head')
