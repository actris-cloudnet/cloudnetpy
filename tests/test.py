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
from tests.run_testcase_processing import process_cloudnetpy


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
        is_file = os.path.isfile(input_path + save_name)
        if not is_file:
            extract_zip(input_path)

    # Call functions to load and storage test data
    c_path = os.path.split(os.getcwd())[0]
    input_path = os.path.join(os.getcwd() + '/source_data/')
    output_path = '/home/korpinen/Documents/ACTRIS/cloudnet_data/'
    url = '/home/korpinen/Documents/ACTRIS/cloudnet_data/test_data_raw.zip'
    load_test_data()

    # Test what inside of raw files
    print('\nCheck all needed variables in raw files')
    pytest.main([c_path + '/cloudnetpy/instruments/tests/raw_files_test.py'])
    #TODO: Break the run if any test fails

    #TODO: jos filut jo olemassa, koko prosessointi tehty ja voidaan ohittaa tämä
    if not os.path.isfile(input_path + 'categorize_file.nc'):
        # processing the CloudnetPy from raw
        print("\nProcessing CloudnetPy from raw files")
        process_cloudnetpy('mace-head', input_path)
        print("\n Done processing test case")
    else:
        print("\nCloudnetPy processing files already exist in directory")

    #TODO: tässä välissä tsekataan kaikki loput filut
    print('\nCheck all needed variables in category file')
    pytest.main([c_path + '/cloudnetpy/tests/categorize_file_test.py'])
    print('\nCheck all needed variables in product files')
    pytest.main([c_path + '/cloudnetpy/products/tests/product_files_test.py'])

if __name__ == "__main__":
    main()





"""
# Define paths to different test files
path = os.path.split(os.getcwd())[0]
product_tests = path + '/cloudnetpy/products/tests/'
category_tests = path + '/cloudnetpy/tests/'
instrument_tests = path + '/cloudnetpy/instruments/tests'

# List of product that is wanted to test
products = ['iwc', 'lwc']

# Run select product tests
for item in products:
    select_product = os.path.join(product_tests, 'test_' + item + '.py')
    pytest.main(['-x', select_product])

# Luodaan optio ajaa halutessaan muita testejä
"""