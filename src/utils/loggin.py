import logging

def setup_logger(name="rag"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger