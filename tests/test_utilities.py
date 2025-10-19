"""
Utility function tests for IRBStudio.

Priority 3: Utilities (9 tests focusing on logging)
- Logging functionality
- Logger configuration
"""

import pytest
import logging
from pathlib import Path

from irbstudio.utils.logging import get_logger


class TestLogging:
    """Test logging utilities."""
    
    def test_get_logger_basic(self):
        """Test that get_logger() returns a logger."""
        logger = get_logger(__name__)
        
        assert logger is not None
        assert isinstance(logger, logging.Logger)
    
    def test_get_logger_with_name(self):
        """Test logger with custom name."""
        logger = get_logger('test_module')
        
        assert logger is not None
        assert logger.name == 'test_module'
    
    def test_get_logger_log_levels(self):
        """Test that logger supports different log levels."""
        logger = get_logger('test_levels')
        
        # Logger should have standard log level methods
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'critical')
        
        # Should be able to call them without error
        logger.debug('Debug message')
        logger.info('Info message')
        logger.warning('Warning message')
        logger.error('Error message')
    
    def test_get_logger_component_specific(self):
        """Test component-specific loggers."""
        logger_sim = get_logger('irbstudio.simulation')
        logger_calc = get_logger('irbstudio.engine')
        
        assert logger_sim.name == 'irbstudio.simulation'
        assert logger_calc.name == 'irbstudio.engine'
        
        # Different loggers should be independent
        assert logger_sim is not logger_calc
    
    def test_get_logger_multiple_calls_same_name(self):
        """Test that multiple calls with same name return same logger."""
        logger1 = get_logger('shared_logger')
        logger2 = get_logger('shared_logger')
        
        # Should be the same logger instance
        assert logger1 is logger2
    
    def test_get_logger_hierarchy(self):
        """Test logger hierarchy."""
        parent_logger = get_logger('irbstudio')
        child_logger = get_logger('irbstudio.simulation')
        
        # Child logger name should start with parent name
        assert child_logger.name.startswith(parent_logger.name)
    
    def test_get_logger_message_logging(self, caplog):
        """Test that messages are actually logged."""
        logger = get_logger('test_logging')
        
        with caplog.at_level(logging.INFO):
            logger.info('Test message')
        
        # Check that message was captured
        assert 'Test message' in caplog.text
    
    def test_get_logger_level_filtering(self, caplog):
        """Test that log level filtering works."""
        logger = get_logger('test_filtering')
        
        # Set to WARNING level, INFO should not appear
        with caplog.at_level(logging.WARNING):
            logger.info('This should not appear')
            logger.warning('This should appear')
        
        assert 'This should not appear' not in caplog.text
        assert 'This should appear' in caplog.text
    
    def test_get_logger_module_tracking(self):
        """Test that logger tracks module name."""
        logger = get_logger(__name__)
        
        # Logger name should be the module name
        assert logger.name == __name__
        
        # Logger should be able to log with context
        logger.info(f"Logging from {__name__}")
