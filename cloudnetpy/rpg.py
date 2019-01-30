import numpy as np
import numpy.ma as ma
from collections import namedtuple


class Rpg:
    """RPG Cloud Radar reader."""
    def __init__(self, filename):
        self.filename = filename
        self.file_position = 0
        self.is_level0 = self.get_file_type()
        self.header = self.read_rpg_header()
        self.dual_pol = self.is_dual_pol()
        self.data = self.read_rpg_data()

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

    def get_file_type(self):
        """Returns True if file is Level 0 file."""
        if self.filename[-3:].lower() == 'lv0':
            return True
        return False

    def is_dual_pol(self):
        if self.header['DualPol'] > 0:
            return True
        return False

    def read_rpg_header(self):

        def insert(names, dtype=np.int32, n_values=1):
            for name in names:
                header[name] = np.fromfile(f, dtype, int(n_values))

        header = {}
        f = open(self.filename, 'rb')

        insert(('FileCode', 'HeaderLen'))
        insert(('StartTime', 'StopTime'), np.uint32)
        insert(('CGProg', 'ModelNo'))

        header['ProgName'] = Rpg.read_string(f)
        header['CustName'] = Rpg.read_string(f)

        insert(('Freq', 'AntSep', 'AntDia', 'AntGain', 'AntBW'), np.float32)

        header['AntGain'] = 10*np.log10(header['AntGain'])

        if self.is_level0:
            insert(('RadarConst',), np.float32)

        insert(('DualPol',), np.int8)

        if self.is_level0:
            insert(('CompEna', 'AntiAlias'), np.int8)

        insert(('SampDur', 'GPSLat', 'GPSLon'), np.float32)
        insert(('CalInt', 'NumbGates', 'NumbLayersT', 'NumbLayersH', 'SequN'))

        insert(('RAlts',), np.float32, header['NumbGates'])
        insert(('TAlts',), np.float32, header['NumbLayersT'])
        insert(('HAlts',), np.float32, header['NumbLayersH'])

        if self.is_level0:
            insert(('RangeFact',), n_values=header['NumbGates'])

        seq_un = header['SequN']
        insert(('SpecN', 'RngOffs', 'ChirpReps'), n_values=seq_un)
        insert(('SeqIntTime', 'dR', 'MaxVel'), np.float32, seq_un)

        if self.is_level0:
            insert(('ChanBW',), np.float32, seq_un)
            insert(('ChirpLowIF', 'ChirpHighIF', 'RangeMin', 'RangeMax',
                    'ChirpFFTSize', 'ChirpInvSmpl'), n_values=seq_un)
            insert(('ChirpCntrFreq', 'ChirpBWFreq'), np.float32, seq_un)
            insert(('FFTStrtInd', 'FFTStopInd', 'ChirpFFTNo'), n_values=seq_un)
            insert(('SampRate', 'MaxRange'))

        insert(('SupPowLev', 'SpkFilEna', 'PhaseCorr', 'RelPowCorr', 'FFTWin'), np.int8)
        insert(('FFTIntRng',))
        insert(('NoiseFilt',), np.float32)

        if self.is_level0:
            insert(('RSV1',), np.int32, 25)
            insert(('RSV2', 'RSV3'), np.uint32, 5000)

        self.file_position = f.tell()
        f.close()
        return header

    def read_rpg_data(self):
        """Read the binary data."""

        Dimensions = namedtuple('Dimensions', ['n_samples',
                                               'n_gates',
                                               'n_layers_t',
                                               'n_layers_h'])

        def create_dimensions():
            n_samples = np.fromfile(f, np.int32, 1)
            return Dimensions(int(n_samples),
                              int(self.header['NumbGates']),
                              int(self.header['NumbLayersT']),
                              int(self.header['NumbLayersH']))

        def create_shapes():
            return((dims.n_samples,),
                   (dims.n_samples, dims.n_layers_t),
                   (dims.n_samples, dims.n_layers_h),
                   (dims.n_samples, dims.n_gates))

        def create_meta():
            """Variable names, dimensions and input/output data types.

            These need to be defined in the same order as they appear in
            the file.

            """
            shapes = create_shapes()
            vars = {}
            vars['SampBytes'] = (shapes[0], np.int32, np.int)
            vars['Time'] = (shapes[0], np.uint32, np.int)
            vars['Time_usec'] = (shapes[0], np.int32, np.int)
            vars['QF'] = (shapes[0], np.int8, np.int)

            for var_name in ('RR', 'RH', 'T', 'P', 'WS', 'WD', 'DD_V', 'Tb',
                             'LWP', 'PowIF', 'El', 'Az', 'BlwStatus',
                             'TransPow', 'TransT', 'RecT', 'PCT'):
                vars[var_name] = (shapes[0], np.float32, np.float)

            vars['T_Prof'] = (shapes[1], np.float32, np.float)

            for var_name in ('AbsHumid_Prof', 'RH_Prof'):
                vars[var_name] = (shapes[2], np.float32, np.float)

            for var_name in ('Sensit_v', 'Sensit_h'):
                vars[var_name] = (shapes[3], np.float32, np.float)

            vars['PrMsk'] = (shapes[3], np.int8, np.int)

            for var_name in ('Zv', 'Vel', 'SW', 'Skew', 'Kurt', 'LDR', 'CorrC',
                             'PhiX'):
                vars[var_name] = (shapes[3], np.float32, np.float)

            return vars

        def create_data():
            """Data dictionary."""
            data_out = {}
            for name in meta:
                shape, _, dtype = meta[name][:]
                data_out[name] = ma.masked_all(shape, dtype=dtype)
            return data_out

        def keyrange(key1, key2):
            """Indices of dict fields from one key to another."""
            all_keys = list(data.keys())
            first_index = all_keys.index(key1)
            second_index = all_keys.index(key2)
            return all_keys[first_index:second_index+1]

        def append(name, n_elements):
            x = np.fromfile(f, meta[name][1], n_elements)
            if n_elements == 1:
                data[name][sample] = x
            else:
                data[name][sample][:] = x

        f = open(self.filename, 'rb')
        f.seek(self.file_position)
        dims = create_dimensions()
        meta = create_meta()
        data = create_data()

        for sample in range(dims.n_samples):
            for key in keyrange('SampBytes', 'PCT'):
                append(key, 1)
            _ = np.fromfile(f, np.int32, 3)
            append('T_Prof', dims.n_layers_t)
            for key in keyrange('AbsHumid_Prof', 'RH_Prof'):
                append(key, dims.n_layers_h)
            for key in keyrange('Sensit_v', 'PrMsk'):
                append(key, dims.n_gates)

            for key, value in data.items():
                print(key,value)

            for gate in range(dims.n_gates):
                if data['PrMsk'][sample][gate] == 1:
                    for key in keyrange('Zv', 'Kurt'):
                        append(key, 1)
                    if self.dual_pol:
                        append('LDR', 1)
                        append('PhiX', 1)
        f.close()
        return data

