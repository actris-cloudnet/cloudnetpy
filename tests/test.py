# Tällä tiedostolla ajetaan kaikki tarpeelliset unit testit halutussa järjestyksessä
# Tämä tiedosto on tarkoitettu lähinnä niille, jotka haluavat kehittää uutta menetelmää jollekin tuotteelle.
# Tähän koodiin voi lisätä muitakin käyttötarkoituksia halutessaan, pitää keskustella

import pytest
import os

def initialize_test_data():
    # Path to test data
    return os.getcwd() + '/'

# Define main path
path = os.path.split(os.getcwd())[0]

# Define paths to different test files
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