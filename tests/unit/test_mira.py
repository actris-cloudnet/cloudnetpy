from cloudnetpy.instruments import mira


class _Object:
    def __init__(self, source):
        self.dataset = _Nest(source)


class _Nest:
    def __init__(self, source):
        self.source = source


def test_find_measurement_date():
    radar = _Object('190517_000002.pds.off')
    date = mira._find_measurement_date(radar)
    assert date == ('2019', '05', '17')
