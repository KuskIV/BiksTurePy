import h5py

def key_to_string(self, key):
    return f"{key[0]}/{key[1]}/{key[2]}"

def get_ppm_arr(h5:h5py._hl.files.File, keys:list, ppm_name:str)->list:
    """Get the ppm array from h5 file

    Args:
        h5 (h5py._hl.files.File): h5py file that should be read
        keys (list): list of keys
        ppm_name (str): the name of the wanted ppm

    Returns:
        list: array represnting the ppm image form the h5py file
    """
    return h5[keys[0]][keys[1]][keys[2]][keys[3]][ppm_name]

def get_key(h5:h5py._hl.files.File, keys:list)->str:
    """Get the key for a image

    Args:
        h5 (h5py._hl.files.File): the h5py file
        keys (list): a list of the keys

    Returns:
        str: the key in string from
    """
    return h5[keys[0]][keys[1]][keys[2]][keys[3]]

def get_linux_constructor():
    return key_to_string, get_ppm_arr, get_key