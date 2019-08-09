# Tällä tiedostolla ajetaan kaikki tarpeelliset unit testit halutussa järjestyksessä
# Tämä tiedosto on tarkoitettu lähinnä niille, jotka haluavat kehittää uutta menetelmää jollekin tuotteelle.
# Tähän koodiin voi lisätä muitakin käyttötarkoituksia halutessaan, pitää keskustella

"""
Tällä koodilla ajetaan koko testi ja prosessointiketju. Erilliset yksittäiset testit ajetaan muualla alustavasti

- Ladataan testi casen raw-data zipinä jostain hakemistosta (lopullinen sijainti päätetään myöhemmin)
- puretaan data valittuun sijaintiin
- Testataan ensiksi raw filut, onko kaikki tarpeellinen sisällä

- Prosessoidaan koko CloudnetPy prosessi

- Testaan instrumentti menetelmät -> testataan tuotettu filu
- Testataan cat menetelmät -> cat filun testaus
- Testataan product menetelmät -> product filujen testaus
- Testataan mahdollisesti vielä plottaus menetelmät, jos tarpeellista
"""

import pytest
import os
import glob
from zipfile import ZipFile
import requests
from tests.run_testcase_processing import *
import warnings

warnings.filterwarnings("ignore")


def main():

    def load_test_data():
        def extract_zip(extract_path):
            # r = requests.get(url)
            # open(save_name, 'wb').write(r.content)
            fl = ZipFile(output_path + save_name, 'r')
            fl.extractall(extract_path)
            fl.close()

        save_name = 'test_data_raw.zip'
        is_dir = os.path.isdir(input_path)
        if not is_dir:
            os.mkdir(input_path )
            extract_zip(input_path)
        else:
            is_file = os.path.isfile(input_path + save_name)
            if not is_file:
                extract_zip(input_path)

    # Call functions to load and storage test data
    c_path = os.path.split(os.getcwd())[0]
    input_path = os.path.join(os.getcwd() + '/source_data/')
    output_path = '/home/korpinen/Documents/ACTRIS/cloudnet_data/'
    url = '/home/korpinen/Documents/ACTRIS/cloudnet_data/test_data_raw.zip'
    load_test_data()

    print('\nCheck all needed variables in raw files')
    pytest.main(["-p no:warnings", c_path + '/cloudnetpy/instruments/tests/raw_files_test.py'])
    #TODO: Break the run if any test fails
    print("\nProcessing CloudnetPy calibrated files from raw files")
    process_cloudnetpy_raw_files('mace-head', input_path)
    print("\n   Done processing test raw files")

    print('\nCheck all needed variables in calibrated files')
    pytest.main(["-p no:warnings", c_path + '/cloudnetpy/instruments/tests/calibrated_files_test.py'])
    print("\nProcessing CloudnetPy categorize file from calibrated files")
    process_cloudnetpy_categorize('mace-head', input_path)
    print("\n   Done processing test categorize file")

    print('\nCheck all needed variables in category file')
    pytest.main(["-p no:warnings", c_path + '/cloudnetpy/tests/categorize_file_test.py'])
    print("\nProcessing CloudnetPy product filea from categorize file")
    process_cloudnetpy_products('mace-head', input_path)
    print("\n   Done processing test product files")

    print('\nCheck all needed variables in product files')
    pytest.main(["-p no:warnings", c_path + '/cloudnetpy/products/tests/product_files_test.py'])

    print("\n###########################################################"
          "\n# All tests passed and processing cloudnetPy done correct #"
          "\n###########################################################")


def initialize_test_data(instrument):
    """
    Finds all file paths and parses wanted files to list
    """
    test_data = glob.glob(os.getcwd() + '/source_data/*.nc')
    print(test_data)
    paths = []
    # TODO: Write this to run faster!
    for inst in instrument:
        for file in test_data:
            if inst in file:
              paths.append(file)
    return paths


if __name__ == "__main__":
    main()
