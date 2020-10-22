import h5py
import platform
import sys
import re

def get_keys(group:str)->list:
    """This method splits a string up into different substrings, each representing a key in the H5PY file 

    Args:
        group (str): the string to split

    Returns:
        list: The split string 
    """
    return re.split('/', group)

def key_to_string_linux(self, key):
    return f"{key[0]}/{key[1]}/{key[2]}"

def get_ppm_arr_linux(h5:h5py._hl.files.File, keys:list, ppm_name:str)->list:
    """Get the ppm array from h5 file

    Args:
        h5 (h5py._hl.files.File): h5py file that should be read
        keys (list): list of keys
        ppm_name (str): the name of the wanted ppm

    Returns:
        list: array represnting the ppm image form the h5py file
    """
    return h5[keys[0]][keys[1]][keys[2]][keys[3]][ppm_name]

def get_key_linux(h5:h5py._hl.files.File, keys:list)->str:
    """Get the key for a image

    Args:
        h5 (h5py._hl.files.File): the h5py file
        keys (list): a list of the keys

    Returns:
        str: the key in string from
    """
    return h5[keys[0]][keys[1]][keys[2]][keys[3]]

def key_to_string_windows(self, key):
    folder_split = key[2].split('\\\\')

    training = folder_split[0]
    img_folder = folder_split[1]

    return f"{key[0]}/{key[1]}/{training}/{img_folder}"

def get_ppm_arr_windows(h5:h5py._hl.files.File, keys:list, ppm_name:str)->list:
    """Get the ppm array from h5 file

    Args:
        h5 (h5py._hl.files.File): h5py file that should be read
        keys (list): list of keys
        ppm_name (str): the name of the wanted ppm

    Returns:
        list: array represnting the ppm image form the h5py file
    """
    return h5[keys[0]][keys[1]][keys[2]][ppm_name]

def get_key_windows(h5:h5py._hl.files.File, keys:list)->str:
    """Get the key for a image

    Args:
        h5 (h5py._hl.files.File): the h5py file
        keys (list): a list of the keys

    Returns:
        str: the key in string from
    """
    return h5[keys[0]][keys[1]][keys[2]]

def get_linux_constructor():
    return 'linux', key_to_string_linux, get_ppm_arr_linux, get_key_linux, get_keys

def get_windows_constructor():
    return 'windows', key_to_string_windows, get_ppm_arr_windows, get_key_windows, get_keys

def get_os_constructor():
    system = platform.system()

    if system == 'Windows':
        return get_windows_constructor()
    elif system == 'Linux':
        return get_linux_constructor()
    else:
        print('This is not a known OS. Please use Linux or Windows.')
        sys.exit()
    