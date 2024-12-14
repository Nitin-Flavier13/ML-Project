import os
import logging  # any execution that happens has to be logged
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logsPath = os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logsPath,exist_ok=True)

LOG_FILE_PATH = os.path.join(logsPath,LOG_FILE)

logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info("Logging has started")
