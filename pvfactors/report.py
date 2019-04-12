"""Module containing example of report builder functions and classes"""
from collections import OrderedDict
import numpy as np


def example_fn_build_report(report, pvarray):
    # Initialize the report
    if report is None:
        list_keys = ['qinc_front', 'qinc_back', 'iso_front', 'iso_back']
        report = OrderedDict({key: [] for key in list_keys})
    # Add elements to the report
    if pvarray is not None:
        pvrow = pvarray.pvrows[1]  # use center pvrow
        report['qinc_front'].append(
            pvrow.front.get_param_weighted('qinc'))
        report['qinc_back'].append(
            pvrow.back.get_param_weighted('qinc'))
        report['iso_front'].append(
            pvrow.front.get_param_weighted('isotropic'))
        report['iso_back'].append(
            pvrow.back.get_param_weighted('isotropic'))
    else:
        # No calculation was performed, because sun was down
        report['qinc_front'].append(np.nan)
        report['qinc_back'].append(np.nan)
        report['iso_front'].append(np.nan)
        report['iso_back'].append(np.nan)

    return report


class ExampleReportBuilder(object):
    """A class is required to build reports when running calculations with
    multiprocessing because of python constraints"""

    @staticmethod
    def build(report, pvarray):
        return example_fn_build_report(report, pvarray)

    @staticmethod
    def merge(reports):
        """Works for dictionary reports"""
        report = reports[0]
        # Merge only if more than 1 report
        if len(reports) > 1:
            keys_report = list(reports[0].keys())
            for other_report in reports[1:]:
                for key in keys_report:
                    report[key] += other_report[key]
        return report
