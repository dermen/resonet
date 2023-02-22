from resonet import arches
from resonet.net import *
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter as arg_formatter
from argparse import Namespace
import pytest

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