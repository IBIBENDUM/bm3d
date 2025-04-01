import logging
import colorlog
from pathlib import Path

def setupLogger(outputDir: Path) -> logging.Logger:
    logger = logging.getLogger("unetTrainLogger")
    logger.setLevel(logging.DEBUG)

    fileFormatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    consoleFormatter = colorlog.ColoredFormatter(
        "%(asctime)s - %(log_color)s%(levelname)s%(reset)s - %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        reset=True,  
        style='%'
    )

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(consoleFormatter)
    logger.addHandler(consoleHandler)

    logFile = outputDir / "train.log"
    fileHandler = logging.FileHandler(logFile)
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(fileFormatter)
    logger.addHandler(fileHandler)

    return logger
    
