from resonet import arches
from resonet.net import *
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter as arg_formatter
from argparse import Namespace
import pytest
import io
import sys
import numpy as np
import logging
from resonet.params import ARCHES, LOSSES

class TestNet:

    def test_get_parser(self):
        parser = get_parser()
        # test with valid input
        parsed = parser.parse_args(['10', 'data.h5','output_dir'])
        assert parsed.ep == 10
        assert parsed.input == 'data.h5'
        assert parsed.outdir == 'output_dir'
        print('test with valid input passed!')

        # test with missing arguments 1
        with pytest.raises(SystemExit):
            parsed = parser.parse_args(['10','data.h5']) # missing 'outdir' argument
        print('# test with missing arguments 1 passed!')

        # test with missing arguments 2
        with pytest.raises(SystemExit):
            parsed = parser.parse_args(['data.h5','output_dir']) # invalid 'ep' argument
        print('# test with missing arguments 2 passed!')

    def test_get_parser_optional_args(self):
        parser = get_parser()
        # test with valid input
        parsed = parser.parse_args(['10', 'data.h5','output_dir','--lr','0.01'])
        assert parsed.ep == 10
        assert parsed.input == 'data.h5'
        assert parsed.outdir == 'output_dir'
        assert parsed.lr == 0.01
        print('# test with valid input passed!')

        # test with invalid input
        with pytest.raises(SystemExit):
            parsed = parser.parse_args(['10', 'data.h5','output_dir','--lr','0.01','--netnum','19'])

    def test_help_option(self):
        # Capture stdout
        stdout = io.StringIO()
        sys.stdout = stdout
        parser = get_parser()
        try:
            with pytest.raises(SystemExit):
                parsed = parser.parse_args(['-h'])
        finally:
            # Reset stdout
            sys.stdout = sys.__stdout__

        output = stdout.getvalue()
        
        # chekcing if the help message contains the following
        assert 'number of epochs' in output
        assert 'input training data h5 file' in output
        assert 'store output files here (will create if necessary)' in output
        assert 'learning rate (important!)' in output


    def test_get_logger(self):
        # Test case 1
        logger = get_logger()
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'resonet'
        assert logger.level == logging.INFO
        print('# test case 1 passed!')

        # Test case 2
        logger = get_logger(filename='test.log', level='debug')
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'resonet'
        assert logger.level == logging.INFO
        print('# test case 2 passed!')

        # Test case 3
        logger = get_logger(do_nothing=True)
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'root'
        assert logger.level == logging.CRITICAL
        print('# test case 3 passed!')
    
    
    