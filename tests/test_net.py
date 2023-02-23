from resonet import arches
from resonet.net import *
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter as arg_formatter
from argparse import Namespace
import pytest

import time
import os
import sys
import h5py
import numpy as np
import logging
from scipy.stats import pearsonr, spearmanr
import pylab as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryJaccardIndex


from resonet.params import ARCHES, LOSSES
from resonet.loaders import H5SimDataDset
from resonet import arches

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
    