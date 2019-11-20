"""Module containing the functions to run engine calculations in normal or
parallel mode."""

from pvfactors import logging
from pvfactors.geometry import OrderedPVArray
from pvfactors.irradiance import HybridPerezOrdered
from pvfactors.viewfactors import VFCalculator
from pvfactors.engine import PVEngine
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from time import time


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def run_timeseries_engine(fn_build_report, pvarray_parameters,
                          timestamps, dni, dhi, solar_zenith, solar_azimuth,
                          surface_tilt, surface_azimuth, albedo,
                          cls_pvarray=OrderedPVArray, cls_engine=PVEngine,
                          cls_irradiance=HybridPerezOrdered,
                          cls_vf=VFCalculator,
                          fast_mode_pvrow_index=None,
                          fast_mode_segment_index=None,
                          irradiance_model_params=None,
                          vf_calculator_params=None,
                          ghi=None):
    """Run timeseries simulation without multiprocessing. This is the
    functional approach to the :py:class:`~pvfactors.engine.PVEngine` class.

    Parameters
    ----------
    fn_build_report : function
        Function that will build the report of the simulation
    pvarray_parameters : dict
        The parameters defining the PV array
    timestamps : array-like
        List of timestamps of the simulation.
    dni : array-like
        Direct normal irradiance values [W/m2]
    dhi : array-like
        Diffuse horizontal irradiance values [W/m2]
    solar_zenith : array-like
        Solar zenith angles [deg]
    solar_azimuth : array-like
        Solar azimuth angles [deg]
    surface_tilt : array-like
        Surface tilt angles, from 0 to 180 [deg]
    surface_azimuth : array-like
        Surface azimuth angles [deg]
    albedo : array-like
        Albedo values (or ground reflectivity)
    cls_pvarray : class of PV array, optional
        Class that will be used to build the PV array
        (Default =
        :py:class:`~pvfactors.geometry.pvarray.OrderedPVArray` class)
    cls_engine : class of PV engine, optional
        Class of the engine to use to run the simulations (Default =
        :py:class:`~pvfactors.engine.PVEngine` class)
    cls_irradiance : class of irradiance model, optional
        The irradiance model that will be applied to the PV array
        (Default =
        :py:class:`~pvfactors.irradiance.models.HybridPerezOrdered` class)
    cls_vf : class of VF calculator, optional
        Calculator that will be used to calculate the view factor matrices
        (Default =
        :py:class:`~pvfactors.viewfactors.calculator.VFCalculator` class)
    fast_mode_pvrow_index : int, optional
        If a valid pvrow index is passed, then the PVEngine fast mode
        will be activated and the engine calculation will be done only
        for the back surface of the selected pvrow (Default = None)
    fast_mode_segment_index : int, optional
        If a segment index is passed, then the PVEngine fast mode
        will calculate back surface irradiance only for the
        selected segment of the selected back surface (Default = None)
    irradiance_model_params : dict, optional
        Dictionary of parameters that will be passed to the irradiance model
        class as kwargs at instantiation (Default = None)
    vf_calculator_params : dict, optional
        Dictionary of parameters that will be passed to the VF calculator
        class as kwargs at instantiation (Default = None)
    ghi : array-like, optional
        Global horizontal irradiance values [W/m2] (Default = None)

    Returns
    -------
    report
        Saved results from the simulation, as specified by user's report
        function
    """

    # Prepare input parameters
    irradiance_model_params = irradiance_model_params or {}
    vf_calculator_params = vf_calculator_params or {}
    # Instantiate classes and engine
    irradiance_model = cls_irradiance(**irradiance_model_params)
    vf_calculator = cls_vf(**vf_calculator_params)
    pvarray = cls_pvarray.init_from_dict(pvarray_parameters)
    eng = cls_engine(pvarray, irradiance_model=irradiance_model,
                     vf_calculator=vf_calculator,
                     fast_mode_pvrow_index=fast_mode_pvrow_index,
                     fast_mode_segment_index=fast_mode_segment_index)
    # Fit engine
    eng.fit(timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
            surface_azimuth, albedo, ghi=ghi)

    # Run all timesteps
    report = (eng.run_full_mode(fn_build_report=fn_build_report)
              if fast_mode_pvrow_index is None
              else eng.run_fast_mode(fn_build_report=fn_build_report))

    return report


def run_parallel_engine(report_builder, pvarray_parameters,
                        timestamps, dni, dhi, solar_zenith, solar_azimuth,
                        surface_tilt, surface_azimuth, albedo,
                        cls_pvarray=OrderedPVArray, cls_engine=PVEngine,
                        cls_irradiance=HybridPerezOrdered,
                        cls_vf=VFCalculator, fast_mode_pvrow_index=None,
                        fast_mode_segment_index=None,
                        irradiance_model_params=None,
                        vf_calculator_params=None, n_processes=2,
                        ghi=None):
    """Run timeseries simulation using multiprocessing. Here, instead of a
    function that will build the report, the users will need to pass a class
    (or an object).

    Parameters
    ----------
    report_builder : class or object
        Class or object that will build and merge the reports. It **must**
        have a ``build()`` and a ``merge()`` method that perform the tasks
    pvarray_parameters : dict
        The parameters defining the PV array
    timestamps : array-like
        List of timestamps of the simulation.
    dni : array-like
        Direct normal irradiance values [W/m2]
    dhi : array-like
        Diffuse horizontal irradiance values [W/m2]
    solar_zenith : array-like
        Solar zenith angles [deg]
    solar_azimuth : array-like
        Solar azimuth angles [deg]
    surface_tilt : array-like
        Surface tilt angles, from 0 to 180 [deg]
    surface_azimuth : array-like
        Surface azimuth angles [deg]
    albedo : array-like
        Albedo values (or ground reflectivity)
    cls_pvarray : class of PV array, optional
        Class that will be used to build the PV array
        (Default =
        :py:class:`~pvfactors.geometry.pvarray.OrderedPVArray` class)
    cls_engine : class of PV engine, optional
        Class of the engine to use to run the simulations (Default =
        :py:class:`~pvfactors.engine.PVEngine` class)
    cls_irradiance : class of irradiance model, optional
        The irradiance model that will be applied to the PV array
        (Default =
        :py:class:`~pvfactors.irradiance.models.HybridPerezOrdered` class)
    cls_vf : class of VF calculator, optional
        Calculator that will be used to calculate the view factor matrices
        (Default =
        :py:class:`~pvfactors.viewfactors.calculator.VFCalculator` class)
    fast_mode_pvrow_index : int, optional
        If a valid pvrow index is passed, then the PVEngine fast mode
        will be activated and the engine calculation will be done only
        for the back surface of the selected pvrow (Default = None)
    fast_mode_segment_index : int, optional
        If a segment index is passed, then the PVEngine fast mode
        will calculate back surface irradiance only for the
        selected segment of the selected back surface (Default = None)
    irradiance_model_params : dict, optional
        Dictionary of parameters that will be passed to the irradiance model
        class as kwargs at instantiation (Default = None)
    vf_calculator_params : dict, optional
        Dictionary of parameters that will be passed to the VF calculator
        class as kwargs at instantiation (Default = None)
    n_processes : int, optional
        Number of parallel processes to run for the calculation (Default = 2)
    ghi : array-like, optional
        Global horizontal irradiance values [W/m2] (Default = None)

    Returns
    -------
    report
        Saved results from the simulation, as specified by user's report
        class (or object)
    """
    # Choose number of workers
    if n_processes == -1:
        n_processes = cpu_count()

    # Make sure albedo is iterable
    if np.isscalar(albedo):
        albedo = albedo * np.ones(len(dni))

    # Prepare class input parameters
    irradiance_model_params = irradiance_model_params or {}
    vf_calculator_params = vf_calculator_params or {}

    # Fix: np.array_split doesn't work well on pd.DatetimeIndex objects
    if isinstance(timestamps, pd.DatetimeIndex):
        timestamps = timestamps.to_pydatetime()

    # Split all arguments according to number of processes
    (folds_timestamps, folds_surface_azimuth, folds_surface_tilt,
     folds_solar_zenith, folds_solar_azimuth, folds_dni, folds_dhi,
     folds_albedo) = map(np.array_split,
                         [timestamps, surface_azimuth, surface_tilt,
                          solar_zenith, solar_azimuth, dni, dhi, albedo],
                         [n_processes] * 8)
    folds_report_builder = [report_builder] * n_processes
    folds_params = [pvarray_parameters] * n_processes
    folds_cls_pvarray = [cls_pvarray] * n_processes
    folds_cls_engine = [cls_engine] * n_processes
    folds_cls_irradiance = [cls_irradiance] * n_processes
    folds_cls_vf = [cls_vf] * n_processes
    folds_fast_mode_pvrow_index = [fast_mode_pvrow_index] * n_processes
    folds_fast_mode_segment_index = [fast_mode_segment_index] * n_processes
    folds_irradiance_model_params = [irradiance_model_params] * n_processes
    folds_vf_calculator_params = [vf_calculator_params] * n_processes
    folds_ghi = ([ghi] * n_processes if ghi is None else
                 np.array_split(ghi, n_processes))
    report_indices = list(range(n_processes))

    # Zip all the folds together
    list_folds = zip(*(folds_report_builder, folds_params, folds_timestamps,
                       folds_dni, folds_dhi, folds_solar_zenith,
                       folds_solar_azimuth, folds_surface_tilt,
                       folds_surface_azimuth, folds_albedo, folds_cls_pvarray,
                       folds_cls_engine, folds_cls_irradiance, folds_cls_vf,
                       folds_fast_mode_pvrow_index,
                       folds_fast_mode_segment_index,
                       folds_irradiance_model_params,
                       folds_vf_calculator_params, folds_ghi,
                       report_indices))

    # Start multiprocessing
    pool = Pool(n_processes)
    start = time()
    indexed_reports = pool.map(_run_serially, list_folds)
    end = time()
    pool.close()
    pool.join()
    sorted_indexed_reports = sorted(indexed_reports, key=lambda tup: tup[1])
    sorted_reports = [tup[0] for tup in sorted_indexed_reports]

    LOGGER.info("Parallel calculation elapsed time: %s sec" % str(end - start))

    report = report_builder.merge(sorted_reports)

    return report


# Utility function for parallel run
def _run_serially(args):
    """Helper function used to run calculations in parallel

    Parameters
    ----------
    args : tuple
        List of arguments where most will be used in
        :py:function:`~pvfactors.run.run_timeseries_engine`

    Returns
    -------
    report
        Saved results from the simulation, as specified by user's report
        class (or object)
    idx : int
        Index of the report, which will be used to sort the final list of
        reports after all the parallel simulations are over

    """
    report_builder, pvarray_parameters, timestamps, dni, dhi, \
        solar_zenith, solar_azimuth, surface_tilt, surface_azimuth,\
        albedo, cls_pvarray, cls_engine, cls_irradiance, cls_vf, \
        fast_mode_pvrow_index, fast_mode_segment_index, \
        irradiance_model_params, vf_calculator_params, ghi, idx = args

    report = run_timeseries_engine(
        report_builder.build, pvarray_parameters,
        timestamps, dni, dhi, solar_zenith, solar_azimuth,
        surface_tilt, surface_azimuth, albedo,
        cls_pvarray=cls_pvarray, cls_engine=cls_engine,
        cls_irradiance=cls_irradiance, cls_vf=cls_vf,
        fast_mode_pvrow_index=fast_mode_pvrow_index,
        fast_mode_segment_index=fast_mode_segment_index,
        irradiance_model_params=irradiance_model_params,
        vf_calculator_params=vf_calculator_params,
        ghi=ghi)

    return report, idx
