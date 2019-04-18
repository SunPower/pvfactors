"""Module containing examples of report builder functions and classes."""

from collections import OrderedDict
import numpy as np


def example_fn_build_report(report, pvarray):
    """Example function that builds a report when used in the
    :py:class:`~pvfactors.engine.PVEngine`. Here it will be a dictionary
    with lists of calculated values.

    Parameters
    ----------
    report : dict
        Initially ``None``, this will be passed and updated by the function
    pvarray : PV array object
        PV array with updated calculation values

    Returns
    -------
    report : dict
        Report updated with newly calculated values
    """
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
        """Method that will build the simulation report. Here we're using the
        previously defined
        :py:function:`~pvfactors.report.example_fn_build_report`.

        Parameters
        ----------
        report : dict
            Initially ``None``, this will be passed and updated by the function
        pvarray : PV array object
            PV array with updated calculation values

        Returns
        -------
        report : dict
            Report updated with newly calculated values
        """
        return example_fn_build_report(report, pvarray)

    @staticmethod
    def merge(reports):
        """Method used to merge multiple reports together. Here it simply
        concatenates the lists of values saved in the different reports.

        Parameters
        ----------
        reports : list of dict
            List of reports that need to be concatenated together

        Returns
        -------
        report : dict
            Final report with all concatenated values
        """
        report = reports[0]
        # Merge only if more than 1 report
        if len(reports) > 1:
            keys_report = list(reports[0].keys())
            for other_report in reports[1:]:
                for key in keys_report:
                    report[key] += other_report[key]
        return report
