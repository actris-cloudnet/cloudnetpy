""" This module contains unit tests for ceilo-module. """
from cloudnetpy.instruments import ceilo
import pytest


def test_find_ceilo_model_jenoptik():
    assert ceilo._find_ceilo_model('ceilo.nc') == 'chm15k'


@pytest.mark.parametrize("fix, result", [
    ('CL01', 'cl51'),
    ('CL02', 'cl31'),
    ('CT02', 'ct25k'),
])
def test_find_ceilo_model_vaisala(fix, result, tmpdir):
    file_name = '/'.join((str(tmpdir), 'ceilo.txt'))
    f = open(file_name, 'w')
    f.write('row\n')
    f.write('\n')
    f.write('row\n')
    f.write(f"-{fix}\n")
    f.close()
    assert ceilo._find_ceilo_model(str(file_name)) == result
