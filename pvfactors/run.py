"""Module containing the functions to run engine calculations in normal or
parallel model"""
from pvfactors.geometry import OrderedPVArray
from pvfactors.irradiance import HybridPerezOrdered
from pvfactors.viewfactors import VFCalculator
from pvfactors.engine import PVEngine


def run_timeseries_engine(fn_build_report, pvarray_parameters,
                          timestamps, dni, dhi, solar_zenith, solar_azimuth,
                          surface_tilt, surface_azimuth, albedo,
                          cls_pvarray=OrderedPVArray, cls_engine=PVEngine,
                          cls_irradiance=HybridPerezOrdered,
                          cls_vf=VFCalculator):

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


def run_parallel_engine():
    pass
