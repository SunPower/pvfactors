"""Module containing examples of report builder functions and classes."""


def example_fn_build_report(pvarray):
    """Example function that builds a report when used in the
    :py:class:`~pvfactors.engine.PVEngine` with full or fast mode simulations.
    Here it will be a dictionary with lists of calculated values.

    Parameters
    ----------
    pvarray : PV array object
        PV array with updated calculation values

    Returns
    -------
    report : dict
        Report updated with newly calculated values
    """
    return {'qinc_front': pvarray.ts_pvrows[1].front
            .get_param_weighted('qinc').tolist(),
            'qinc_back': pvarray.ts_pvrows[1].back
            .get_param_weighted('qinc').tolist(),
            'iso_front': pvarray.ts_pvrows[1].front
            .get_param_weighted('isotropic').tolist(),
            'iso_back': pvarray.ts_pvrows[1].back
            .get_param_weighted('isotropic').tolist()}


class ExampleReportBuilder(object):
    """A class is required to build reports when running calculations with
    multiprocessing because of python constraints"""

    @staticmethod
    def build(pvarray):
        """Method that will build the simulation report. Here we're using the
        previously defined
        :py:function:`~pvfactors.report.example_fn_build_report`.

        Parameters
        ----------
        pvarray : PV array object
            PV array with updated calculation values

        Returns
        -------
        report : dict
            Report updated with newly calculated values
        """
        return example_fn_build_report(pvarray)

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
        # merge other reports if any
        keys_report = list(reports[0].keys())
        for other_report in reports[1:]:
            for key in keys_report:
                report[key] += other_report[key]
        return report
