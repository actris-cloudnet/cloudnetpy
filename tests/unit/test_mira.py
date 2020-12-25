import pytest
from cloudnetpy.instruments import mira


class TestMeasurementDate:

    correct_date = ['2020', '05', '24']

    @pytest.fixture(autouse=True)
    def _init(self, raw_mira_file):
        self.raw_radar = mira.Mira(raw_mira_file, {'name': 'Test'})

    def test_validate_date_default(self):
        self.raw_radar.validate_date(None)
        assert self.raw_radar.date == self.correct_date

    def test_validate_date(self):
        self.raw_radar.validate_date('2020-05-24')
        assert self.raw_radar.date == self.correct_date

    def test_validate_date_fails(self):
        with pytest.raises(ValueError):
            self.raw_radar.validate_date('2020-05-23')
