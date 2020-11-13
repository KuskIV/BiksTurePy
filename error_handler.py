import os,sys,inspect

class Error_handler():
    def __init__(self,filepath:str,class_name:str = None):
        """initialise the error handler.

        Args:
            filepath (string): file path is given either as a string or __file__ which is prefered
            class_name (string, optional): class name is given either as a string or __class__.__name__ which is prefered. Defaults to None.
        """
        self.filepath = filepath
        self.class_name = class_name

    def check_if_valid_path(self, path:str, method_name:str):
        """checks if a path exsists in the system, returns an unaltered path if True, exits if false

        Args:
            path (string): the path of the file that should exsist
            method_name (string): method names should be given as inspect.stack()[0][3] or a string.

        Returns:
            str: returns a unalterd path
        """
        if not os.path.exists(path):
            print("----------------------BEGIN_ERROR_MESSAGE----------------------")
            print(f"The path:{path} dosen't seem to exsist")
            print(f"FILE: The error occurred in {self.filepath}")
            if not class_name:
                print(f"CLASS: The class where the error occurred is {self.class_name}")
            print(f"METHOD: The method the errors occurred in is {method_name}")
            print("----------------------END_ERROR_MESSAGE----------------------")
            sys.exit()
        return path

    def custom_error_check(self,error_method:bool, msg:str, method_name:str = None):
        """checks if some error method returns true the message will be printet torgeter with some aditional information

        Args:
            error_method (bool): method that check if an error occurred
            msg (str): message that should be printet
            method_name (str, optional): method names should be given as inspect.stack()[0][3] or a string. Defaults to None.
        """
        if not error_method:
            print("----------------------BEGIN_ERROR_MESSAGE----------------------")
            print(msg)
            print(f"FILE: The error occurred in {self.filepath}")
            if not class_name:
                print(f"CLASS: The class where the error occurred is {self.class_name}")            
            if not method_name == None:
                print(f"METHOD: The method the errors occurred in is {method_name}")
            print("----------------------END_ERROR_MESSAGE----------------------")
