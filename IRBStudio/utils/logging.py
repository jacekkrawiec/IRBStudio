"""Centralized logging configuration for the IRBStudio package."""

import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger with a standardized format.

    Args:
        name: The name for the logger, typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if logger is already configured
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    # Create a handler to write to standard output
    handler = logging.StreamHandler(sys.stdout)
    
    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(handler)
    
    return logger
