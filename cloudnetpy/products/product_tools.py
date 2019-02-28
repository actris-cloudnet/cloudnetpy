import cloudnetpy.utils as utils
"""
Lisätään tänne funktioita, jotka toistuvat useissa moduuleissa.
Yritetään pitää ne suht geneerisinä
"""

def check_active_bits(cb, keys):
    """
    Check is observed bin active or not, returns boolean array of
    active and unactive bin index
    """
    bits = {}
    for i, key in enumerate(keys):
        bits[key] = utils.isbit(cb, i)
    return bits

def get_categorize_keys():
    return ('droplet', 'falling', 'cold',
            'melting', 'aerosol', 'insect')

def get_status_keys():
    return ('radar', 'lidar', 'clutter',
            'molecular', 'attenuated', 'corrected')