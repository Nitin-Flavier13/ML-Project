import sys 
from src.logger import logging

def error_message_detail(error,error_detail: sys):
    # The error_detail.exc_info() function call retrieves information about the 
    # most recent exception caught in a try...except block.
    # exception traceback contains detail info on where the exception happened

    exc_type,exc_value,exc_tb = error_detail.exc_info()

    lineNumber = exc_tb.tb_lineno
    fileName = exc_tb.tb_frame.f_code.co_filename 
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        fileName,lineNumber,str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail: sys):
        super().__init__(error_message)
        self.final_error_message = error_message_detail(error_message,error_detail=error_detail)

    def __str__(self): 
        return self.final_error_message         # print (obj) will print the values

if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Divided by zero")
        raise CustomException(e,sys)
