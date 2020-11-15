import os,sys,inspect


def get_file_location():
    frame = inspect.stack()[7]
    module = inspect.getmodule(frame[0])
    filename = module.__file__
    return filename

def check_if_valid_path(self, path:str, class_name = None):
    """checks if a path exsists in the system, returns an unaltered path if True, exits if false

    Args:
        path (string): the path of the file that should exsist
        method_name (string): method names should be given as inspect.stack()[0][3] or a string.

    Returns:
        str: returns a unalterd path
    """
    if not os.path.exists(path, class_name = None):
        print("----------------------BEGIN_ERROR_MESSAGE----------------------")
        print(f"The path:{path} dosen't seem to exsist")
        print(f"FILE: The error occurred in {get_file_location()}")
        if not class_name == None:
            print(f"CLASS: The class where the error occurred is {self.class_name}")
        print(f"METHOD: The method the errors occurred in is {inspect.stack()[1][3]}")
        print("----------------------END_ERROR_MESSAGE----------------------")
        sys.exit()
    return path

def custom_error_check(self,error_method:bool, msg:str, class_name = None):
    """checks if some error method returns true the message will be printet torgeter with some aditional information

    Args:
        error_method (bool): method that check if an error occurred
        msg (str): message that should be printet
        method_name (str, optional): method names should be given as inspect.stack()[0][3] or a string. Defaults to None.
    """
    if not error_method:
        print("----------------------BEGIN_ERROR_MESSAGE----------------------")
        print(msg)
        print(f"FILE: The error occurred in {get_file_location()}")
        if not class_name == None:
            print(f"CLASS: The class where the error occurred is {self.class_name}")            
        print(f"METHOD: The method the errors occurred in is {inspect.stack()[1][3]}")
        print("----------------------END_ERROR_MESSAGE----------------------")
