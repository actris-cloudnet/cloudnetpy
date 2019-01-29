import numpy as np


class Rpg:
    """RGB Cloud Radar reader."""
    def __init__(self, filename):
        self.filename = filename
        self.header = self.read_rpg_header(filename)

    @staticmethod
    def read_string(file_id):
        """Read characters from binary data until whitespace."""
        str_out = ''
        while True:
            c = np.fromfile(file_id, np.int8, 1)
            if c:
                str_out += chr(c)
            else:
                break
        return str_out

    @staticmethod
    def get_file_type(filename):
        """Returns True if file is Level 0 file."""
        if filename[-3:].lower() == 'lv0':
            return True
        return False

    @staticmethod
    def read_rpg_header(filename):

        def insert(names, dtype=np.int32, n_values=1):
            for name in names:
                header[name] = np.fromfile(f, dtype, int(n_values))

        header = {}
        is_level0 = Rpg.get_file_type(filename)
        f = open(filename, 'rb')
        insert(('FileCode', 'HeaderLen'))
        insert(('StartTime', 'StopTime'), np.uint32)
        insert(('CGProg', 'ModelNo'))
        header['ProgName'] = Rpg.read_string(f)
        header['CustName'] = Rpg.read_string(f)
        insert(('Freq', 'AntSep', 'AntDia', 'AntGain', 'AntBW'), np.float32)
        header['AntGain'] = 10*np.log10(header['AntGain'])
        if is_level0:
            insert(('RadarConst',), np.float32)
        insert(('DualPol',), np.int8)
        if is_level0:
            insert(('CompEna', 'AntiAlias'), np.int8)
        insert(('SampDur', 'GPSLat', 'GPSLon'), np.float32)
        insert(('CalInt', 'NumbGates', 'NumbLayersT', 'NumbLayersH', 'SequN'))
        insert(('RAlts',), np.float32, header['NumbGates'])
        insert(('TAlts',), np.float32, header['NumbLayersT'])
        insert(('HAlts',), np.float32, header['NumbLayersH'])
        if is_level0:
            insert(('RangeFact',), n_values=header['NumbGates'])
        seq_un = header['SequN']
        insert(('SpecN', 'RngOffs', 'ChirpReps'), n_values=seq_un)
        insert(('SeqIntTime', 'dR', 'MaxVel'), np.float32, seq_un)
        if is_level0:
            insert(('ChanBW',), np.float32, seq_un)
            insert(('ChirpLowIF', 'ChirpHighIF', 'RangeMin', 'RangeMax',
                    'ChirpFFTSize', 'ChirpInvSmpl'), n_values=seq_un)
            insert(('ChirpCntrFreq', 'ChirpBWFreq'), np.float32, seq_un)
            insert(('FFTStrtInd', 'FFTStopInd', 'ChirpFFTNo'), n_values=seq_un)
            insert(('SampRate', 'MaxRange'))
        insert(('SupPowLev', 'SpkFilEna', 'PhaseCorr', 'RelPowCorr', 'FFTWin'),
               np.int8)
        insert(('FFTIntRng',))
        insert(('NoiseFilt',), np.float32)
        if is_level0:
            insert(('RSV1',), np.int32, 25)
            insert(('RSV2', 'RSV3'), np.uint32, 5000)
        f.close()
        return header

