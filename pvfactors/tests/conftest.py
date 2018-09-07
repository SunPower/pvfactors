# -*- coding: utf-8 -*-

import pytest
import os
import pandas as pd

DIR_TEST = os.path.dirname(__file__)
DIR_TEST_DATA = os.path.join(DIR_TEST, 'test_files')


@pytest.fixture(scope='function')
def df_outputs():
    """ Example of df_outputs to be used for tests """
    fp = os.path.join(DIR_TEST_DATA, 'file_test_df_outputs.csv')
    df_outputs = pd.read_csv(fp, header=[0, 1, 2], index_col=0)
    df_outputs.index = pd.to_datetime(df_outputs.index)

    yield df_outputs


@pytest.fixture(scope='function')
def df_registries():
    """ Example of df_registries to be used for tests """
    fp = os.path.join(DIR_TEST_DATA, 'file_test_df_registries.csv')
    df_registries = pd.read_csv(fp, header=[0], parse_dates=['timestamps'])
    yield df_registries
