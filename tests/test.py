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
#from bin.process_all import main as CNP_run


def initialize_test_data(instrument):
    """
    Finds all file paths and parses wanted files to list
    """
    test_data = glob.glob(os.getcwd() + '/*.nc')
    paths = []
    # TODO: Write this to run faster!
    for inst in instrument:
        for file in test_data:
            if inst in file:
              paths.append(file)
    return paths


def main():

    def load_test_data():
        """
        Read test data, if test data doesn't exist in dir
        Unpack the raw data file and save data to dir
        return path to files / list of file paths
        Maybe better to return dict of file paths
        ... Lets write this and think what to do at the same time.
        """
        save_name = 'test_data_raw.zip'
        is_file = os.path.isfile(input_path + '/' + save_name)
        if not is_file:
            #r = requests.get(url)
            #open(save_name, 'wb').write(r.content)
            fl = ZipFile(output_path + save_name, 'r')
            fl.extractall(input_path)
            fl.close()

    # Call functions to load and storage test data
    c_path = os.path.split(os.getcwd())[0]
    input_path = os.getcwd()
    output_path = '/home/korpinen/Documents/ACTRIS/cloudnet_data/'
    url = '/home/korpinen/Documents/ACTRIS/cloudnet_data/test_data_raw.zip'
    load_test_data()

    # Test what inside of raw files
    print('\nCheck all needed variables in raw files')
    pytest.main([c_path + '/cloudnetpy/instruments/tests'])

    #TODO: Fall down the run if any test fails

    # processing the CloudnetPy from raw
    print("\nProcessing CloudnetPy from raw files")
    #CNP_run()

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