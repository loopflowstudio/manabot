# manabot/infra/log.py

import logging

def getLogger(name: str) -> logging.Logger:
    """
    Returns a logger using our custom CategoryLogger configuration.
    """
    logger = logging.getLogger(name)
    
    # If the logger hasn't been configured yet, set it up.
    if not getattr(logger, "_custom_configured", False):
        custom_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(custom_format)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.propagate = False
        setattr(logger, "_custom_configured", True)
    return logger

def setGlobalLogLevel(level: int) -> None:
    """
    Set the global log level for all loggers.
    """
    logging.getLogger("manabot").setLevel(level)