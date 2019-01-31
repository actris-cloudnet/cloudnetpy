from collections import namedtuple
import numpy as np
import numpy.ma as ma


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
        """Reads the header or rpg binary file."""
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
        """Read the actual data from rpg binary file."""
        Dimensions = namedtuple('Dimensions', ['n_samples',
                                               'n_gates',
                                               'n_layers_t',
                                               'n_layers_h'])

        def create_dimensions():
            """Loop lengths in the data read."""
            n_samples = np.fromfile(f, np.int32, 1)
            return Dimensions(int(n_samples),
                              int(self.header['NumbGates']),
                              int(self.header['NumbLayersT']),
                              int(self.header['NumbLayersH']))

        def create_shapes():
            """Possible shapes of the data arrays."""
            return((dims.n_samples,),
                   (dims.n_samples, dims.n_layers_t),
                   (dims.n_samples, dims.n_layers_h),
                   (dims.n_samples, dims.n_gates))

        def create_variables():
            """Variable names, data arrays and input data types.

            These need to be defined in the same order as they appear in
            the file.

            """
            shapes = create_shapes()
            fun = np.zeros
            vrs = {}
            vrs['SampBytes'] = (fun(shapes[0], np.int), np.int32)
            vrs['Time'] = (fun(shapes[0], np.int), np.uint32)
            vrs['Time_usec'] = (fun(shapes[0], np.int), np.int32)
            vrs['QF'] = (fun(shapes[0], np.int), np.int8)
            for var_name in ('RR', 'RH', 'T', 'P', 'WS', 'WD', 'DD_V', 'Tb',
                             'LWP', 'PowIF', 'El', 'Az', 'BlwStatus',
                             'TransPow', 'TransT', 'RecT', 'PCT'):
                vrs[var_name] = (fun(shapes[0]), np.float32)
            vrs['T_Prof'] = (fun(shapes[1]), np.float32)
            for var_name in ('AbsHumid_Prof', 'RH_Prof'):
                vrs[var_name] = (fun(shapes[2]), np.float32)
            for var_name in ('Sensit_v', 'Sensit_h'):
                vrs[var_name] = (fun(shapes[3]), np.float32)
            vrs['PrMsk'] = (fun(shapes[3], np.int), np.int8)
            for var_name in ('Zv', 'Vel', 'SW', 'Skew', 'Kurt', 'LDR', 'CorrC',
                             'PhiX'):
                vrs[var_name] = (fun(shapes[3]), np.float32)
            return vrs

        def append(name, n_elements):
            """Append data into already allocated arrays."""
            array, dtype = data[name]
            values = np.fromfile(f, dtype, n_elements)
            if n_elements == 1 and array.ndim == 1:
                array[sample] = values
            elif n_elements == 1 and array.ndim == 2:
                array[sample][gate] = values
            else:
                array[sample][:] = values

        def get_keyranges():
            """Returns dict-names for the different 'groups' of variables.

            The variables are grouped in the binary file into 5 groups.
            The keyranges make it easy to separate these groups once
            you know the first and last variable name in each group.

            """
            def _keyrange(key1, key2):
                """List of keys from one key to another."""
                ind1 = keys.index(key1)
                ind2 = keys.index(key2)
                return keys[ind1:ind2 + 1]

            keys = list(data.keys())
            return (_keyrange('SampBytes', 'PCT'),
                    _keyrange('AbsHumid_Prof', 'RH_Prof'),
                    _keyrange('Sensit_v', 'PrMsk'),
                    _keyrange('Zv', 'Kurt'),
                    _keyrange('LDR', 'PhiX'))

        def _fix_output():
            """Returns just the data arrays as MaskedArrays."""
            out = {}
            for name in data:
                out[name] = ma.masked_equal(data[name][0], 0)
            return out

        f = open(self.filename, 'rb')
        f.seek(self.file_position)
        dims = create_dimensions()
        data = create_variables()
        keyranges = get_keyranges()

        for sample in range(dims.n_samples):

            for key in keyranges[0]:
                append(key, 1)

            _ = np.fromfile(f, np.int32, 3)

            append('T_Prof', dims.n_layers_t)

            for key in keyranges[1]:
                append(key, dims.n_layers_h)

            for key in keyranges[2]:
                append(key, dims.n_gates)

            for gate in range(dims.n_gates):

                if data['PrMsk'][0][sample][gate] == 1:
                    for key in keyranges[3]:
                        append(key, 1)

                    if self.dual_pol:
                        for key in keyranges[4]:
                            append(key, 1)
        f.close()
        return _fix_output()

