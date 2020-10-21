import platform

from .os_linux import get_linux_constructor

def get_os_constructor():
    system = platform.system()

    if system == 'Windows':
        print("This feature does not work for windows as of yet. Please use linux")
    elif system == 'Linux':
        return get_linux_constructor()
    else:
        print('This is not a known OS.')
    