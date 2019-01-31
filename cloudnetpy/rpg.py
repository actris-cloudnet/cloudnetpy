"""This module contains RPG Cloud Radar related functions."""
import os
from collections import namedtuple
import numpy as np
import numpy.ma as ma


class Rpg:
    """RPG Cloud Radar Level 1 data reader."""
    def __init__(self, filename):
        self.filename = filename
        self._file_position = 0
        self._dual_pol = False
        self.header = self.read_rpg_header()
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

    def read_rpg_header(self):
        """Reads the header or rpg binary file."""

        def append(names, dtype=np.int32, n_values=1):
            """Updates header dictionary."""
            for name in names:
                header[name] = np.fromfile(file, dtype, int(n_values))

        header = {}
        file = open(self.filename, 'rb')
        append(('FileCode', 'HeaderLen'))
        append(('StartTime', 'StopTime'), np.uint32)
        append(('CGProg', 'ModelNo'))
        header['ProgName'] = Rpg.read_string(file)
        header['CustName'] = Rpg.read_string(file)
        append(('Freq', 'AntSep', 'AntDia', 'AntGain', 'AntBW'), np.float32)
        header['AntGain'] = 10*np.log10(header['AntGain'])
        append(('DualPol',), np.int8)
        append(('SampDur', 'GPSLat', 'GPSLon'), np.float32)
        append(('CalInt', 'NumbGates', 'NumbLayersT', 'NumbLayersH', 'SequN'))
        append(('RAlts',), np.float32, header['NumbGates'])
        append(('TAlts',), np.float32, header['NumbLayersT'])
        append(('HAlts',), np.float32, header['NumbLayersH'])
        append(('SpecN', 'RngOffs', 'ChirpReps'), n_values=header['SequN'])
        append(('SeqIntTime', 'dR', 'MaxVel'), np.float32, header['SequN'])
        append(('SupPowLev', 'SpkFilEna', 'PhaseCorr', 'RelPowCorr', 'FFTWin'),
               np.int8)
        append(('FFTIntRng',))
        append(('NoiseFilt',), np.float32)
        self._file_position = file.tell()
        file.close()
        if header['DualPol'] > 0:
            self._dual_pol = True
        return header

    def read_rpg_data(self):
        """Reads the actual data from rpg binary file."""
        Dimensions = namedtuple('Dimensions', ['n_samples',
                                               'n_gates',
                                               'n_layers_t',
                                               'n_layers_h'])

        def create_dimensions():
            """Returns loop lengths for the data read."""
            n_samples = np.fromfile(file, np.int32, 1)
            return Dimensions(int(n_samples),
                              int(self.header['NumbGates']),
                              int(self.header['NumbLayersT']),
                              int(self.header['NumbLayersH']))

        def create_shapes():
            """Returns possible shapes of the data arrays."""
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
            # Variable group 0
            vrs['SampBytes'] = (fun(shapes[0], np.int), np.int32)
            vrs['Time'] = (fun(shapes[0], np.int), np.uint32)
            vrs['Time_usec'] = (fun(shapes[0], np.int), np.int32)
            vrs['QF'] = (fun(shapes[0], np.int), np.int8)
            # Variable group 1
            for var_name in ('RR', 'RH', 'T', 'P', 'WS', 'WD', 'DD_V', 'Tb',
                             'LWP', 'PowIF', 'El', 'Az', 'BlwStatus',
                             'TransPow', 'TransT', 'RecT', 'PCT'):
                vrs[var_name] = (fun(shapes[0]), np.float32)
            vrs['T_Prof'] = (fun(shapes[1]), np.float32)
            for var_name in ('AbsHumid_Prof', 'RH_Prof'):
                vrs[var_name] = (fun(shapes[2]), np.float32)
            # Variable group 2
            for var_name in ('Sensit_v', 'Sensit_h'):
                vrs[var_name] = (fun(shapes[3]), np.float32)
            vrs['PrMsk'] = (fun(shapes[3], np.int), np.int8)
            # Variable groups 3 and 4
            for var_name in ('Zv', 'Vel', 'SW', 'Skew', 'Kurt',   # group 3
                             'LDR', 'CorrC', 'PhiX'):             # group 4
                vrs[var_name] = (fun(shapes[3]), np.float32)
            return vrs

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

        def append(name, n_elements):
            """Append data into already allocated arrays."""
            array, dtype = data[name]
            values = np.fromfile(file, dtype, n_elements)
            if n_elements == 1 and array.ndim == 1:
                array[sample] = values
            elif n_elements == 1 and array.ndim == 2:
                array[sample][gate] = values
            else:
                array[sample][:] = values

        def _fix_output():
            """Returns just the data arrays as MaskedArrays."""
            out = {}
            for name in data:
                out[name] = ma.masked_equal(data[name][0], 0)
            return out

        file = open(self.filename, 'rb')
        file.seek(self._file_position)
        dims = create_dimensions()
        data = create_variables()
        keyranges = get_keyranges()

        for sample in range(dims.n_samples):

            for key in keyranges[0]:
                append(key, 1)

            _ = np.fromfile(file, np.int32, 3)

            append('T_Prof', dims.n_layers_t)

            for key in keyranges[1]:
                append(key, dims.n_layers_h)

            for key in keyranges[2]:
                append(key, dims.n_gates)

            for gate in range(dims.n_gates):

                if data['PrMsk'][0][sample][gate] == 1:
                    for key in keyranges[3]:
                        append(key, 1)

                    if self._dual_pol:
                        for key in keyranges[4]:
                            append(key, 1)
        file.close()
        return _fix_output()


def get_rpg_files(path_to_l1_files):
    """Returns list of RPG Level 1 files for one day - sorted by filename."""
    files = os.listdir(path_to_l1_files)
    l1_files = [path_to_l1_files+file for file in files if file.endswith('LV1')]
    l1_files.sort()
    return l1_files


def get_rpg_objects(rpg_files):
    """Creates a list of Rpg() objects from the filenames."""
    for file in rpg_files:
        yield Rpg(file)


def concatenate_rpg_data(rpg_objects):
    """Combines data from hourly Rpg() objects."""
    fields = ('Time', 'Zv', 'LDR', 'CorrC', 'PhiX', 'SW', 'Skew',
              'Kurt', 'Vel', 'LWP', 'T', 'RH', 'TransPow')
    radar = dict.fromkeys(fields, np.array([]))
    for rpg in rpg_objects:
        for name in fields:
            radar[name] = (np.concatenate((radar[name], rpg.data[name]))
                           if radar[name].size else rpg.data[name])
    return radar
