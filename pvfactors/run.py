"""Module containing the functions to run engine calculations in normal or
parallel model"""
from pvfactors import logging
from pvfactors.geometry import OrderedPVArray
from pvfactors.irradiance import HybridPerezOrdered
from pvfactors.viewfactors import VFCalculator
from pvfactors.engine import PVEngine
from multiprocessing import Pool, cpu_count
import numpy as np
from time import time


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def run_timeseries_engine(fn_build_report, pvarray_parameters,
                          timestamps, dni, dhi, solar_zenith, solar_azimuth,
                          surface_tilt, surface_azimuth, albedo,
                          cls_pvarray=OrderedPVArray, cls_engine=PVEngine,
                          cls_irradiance=HybridPerezOrdered,
                          cls_vf=VFCalculator):
    """Run timeseries simulation using the provided pvfactors classes"""

    # Instantiate classes and engine
    irradiance_model = cls_irradiance()
    vf_calculator = cls_vf()
    eng = cls_engine(pvarray_parameters, cls_pvarray=cls_pvarray,
                     irradiance_model=irradiance_model,
                     vf_calculator=vf_calculator)

    # Fit engine
    eng.fit(timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
            surface_azimuth, albedo)

    # Run all timesteps
    report = eng.run_all_timesteps(fn_build_report=fn_build_report)

    return report


def run_parallel_engine(report_builder, pvarray_parameters,
                        timestamps, dni, dhi, solar_zenith, solar_azimuth,
                        surface_tilt, surface_azimuth, albedo,
                        cls_pvarray=OrderedPVArray, cls_engine=PVEngine,
                        cls_irradiance=HybridPerezOrdered,
                        cls_vf=VFCalculator, n_processes=2):
    """Run timeseries simulation using multiprocessing"""
    # Choose number of workers
    if n_processes == -1:
        n_processes = cpu_count()

    # Make sure albedo is iterable
    if np.isscalar(albedo):
        albedo = albedo * np.ones(len(dni))

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
    report_indices = list(range(n_processes))

    # Zip all the folds together
    list_folds = zip(*(folds_report_builder, folds_params, folds_timestamps,
                       folds_dni, folds_dhi, folds_solar_zenith,
                       folds_solar_azimuth, folds_surface_tilt,
                       folds_surface_azimuth, folds_albedo, folds_cls_pvarray,
                       folds_cls_engine, folds_cls_irradiance, folds_cls_vf,
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
    """Helper function used to run calculations in parallel"""
    report_builder, pvarray_parameters, timestamps, dni, dhi, \
        solar_zenith, solar_azimuth, surface_tilt, surface_azimuth,\
        albedo, cls_pvarray, cls_engine, cls_irradiance, cls_vf, idx = args

    report = run_timeseries_engine(
        report_builder.build, pvarray_parameters,
        timestamps, dni, dhi, solar_zenith, solar_azimuth,
        surface_tilt, surface_azimuth, albedo,
        cls_pvarray=cls_pvarray, cls_engine=cls_engine,
        cls_irradiance=cls_irradiance, cls_vf=cls_vf)

    return report, idx
