import sys 


class CustomException(Exception):
    def __init__(self, error_message, error_detials:sys):
        self.error_message = error_message
        _, _, exc_tb = error_detials.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.file_name  = exc_tb.tb_frame.f_code.co_filename

    def __str__(self) -> str:
        """
        Return a custom error message that specifies the file name, 
        line and the error message
        """
        return f"Error occured in python script name [{self.file_name}] line number [{self.lineno}] error message [{str(self.error_message)}]"


if __name__ == "__main__":
    try: 
        pass 
    except Exception as e: 
        raise CustomException(e, sys)