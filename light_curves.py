import matplotlib.pyplot as plt
from lsst.rsp import get_tap_service
import numpy as np
import imageio
import pandas as pd
import os
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from matplotlib.patches import Circle
import time
import gc
import zipfile
import sys

from lsst.daf.butler import Butler
from lsst.daf.butler.registry import ConflictingDefinitionError
import lsst.daf.base as dafBase

import lsst.afw.display as afwDisplay
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.sphgeom

from lsst.source.injection import ingest_injection_catalog, generate_injection_catalog
from lsst.source.injection import VisitInjectConfig, VisitInjectTask

from lsst.pipe.tasks.registerImage import RegisterConfig, RegisterTask
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask

from lsst.meas.algorithms.detection import SourceDetectionTask
from lsst.meas.deblender import SourceDeblendTask
from lsst.meas.base import SingleFrameMeasurementTask
from lsst.meas.base import ForcedMeasurementTask

class LightCurve:
    def __init__ (self, ra=None, dec=None, band="i", data=None, name=None):
        '''data = pd.DataFrame(columns=["Time", "mag", "mag_err", "calexp_detector", "calexp_visit"])'''
        self.ra = ra
        self.dec = dec
        self.htm_id = None
        self.band = band
        if data is None:
            self.data = pd.DataFrame(columns=["Time", "mag", "mag_err", "calexp_detector", "calexp_visit"])
        else:
            self.data = data
        self.name = name 

    def calculate_htm_id(self, level=20):
        pixelization = lsst.sphgeom.HtmPixelization(level)
        htm_id = pixelization.index(
            lsst.sphgeom.UnitVector3d(
                lsst.sphgeom.LonLat.fromDegrees(self.ra, self.dec)))
        circle = pixelization.triangle(htm_id).getBoundingCircle()
        scale = circle.getOpeningAngle().asDegrees()*3600
        level = pixelization.getLevel()
        print("{:<20}".format(f'({self.ra}, {self.dec})') + f'HTM ID={htm_id} at level={level} is bounded by a circle of radius ~{scale:0.2f} arcsec.')
        self.htm_id = htm_id
        return htm_id

    def collect_calexp(self, level=20):
        if not isinstance(self.htm_id, int):
            self.calculate_htm_id(level)
        datasetRefs = list(butler.registry.queryDatasets(
            "calexp", htm20=self.htm_id, where=f"band = '{self.band}'"))
        print("{:<20}".format("") + f"Found {len(datasetRefs)} calexps")
        ccd_visit = butler.get('ccdVisitTable')
        times = []
        detectors = [] ; visits = []
        mags = []
        mag_errs = []

        for calexp_data in datasetRefs:
            did = calexp_data.dataId
            ccdrow = (ccd_visit['visitId'] == did['visit']) & (
                ccd_visit['detector'] == did['detector'])
            exp_midpoint = ccd_visit[ccdrow]['expMidptMJD'].values[0]
            times.append(exp_midpoint)
            detectors.append(did['detector'])  
            visits.append(did['visit'])          
            mags.append(np.nan) ; mag_errs.append(np.nan)  
            
        new_data = pd.DataFrame({
            "Time": times,
            "mag": mags,
            "mag_err": mag_errs,
            "calexp_detector": detectors,
            "calexp_visit": visits})

        new_data = new_data.dropna(axis=1, how='all') # Excluir columnas con todos los valores NA antes de la concatenación
        self.data = pd.concat([self.data, new_data], ignore_index=True)
        self.datasetRefs = datasetRefs  

    def simulate(self, params, model="Pacz", plot=False):
        """
        Simulate magnitudes for a microlensing event and optionally plot the result.
    
        Parameters:
        params (dict): Dictionary containing parameters like t_0, t_E, u_0, and m_base.
        model (str): The model to use for simulation. Default is "Pacz".
        plot (bool): If True, plot the magnification curve.
        
        Returns:
        np.ndarray: Magnitudes corresponding to the times in self.data["Time"].
        """
    
        if model == "Pacz":  # params = {t_0, t_E, u_0, m_base}
            
            def Pacz(t, t_0, t_E, u_0, m_base):
                u_t = np.sqrt(u_0**2 + ((t - t_0) / t_E)**2)
                A_t = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))
                return m_base - 2.5 * np.log10(A_t)
            m_t = Pacz(self.data["Time"], **params)
            self.data["mag"] = m_t
            self.name = "Simulated - Pacz"
        else:
            raise ValueError("Model not recognized. Currently supported models: 'Pacz'.")
    
        if plot:
            t_plot = np.linspace(np.min(self.data["Time"]), np.max(self.data["Time"]), 1000)
            m_plot = Pacz(t_plot, **params)
            plt.plot(t_plot, m_plot, color='gray')
            plt.scatter(self.data["Time"], m_t, color='red')
            plt.xlabel('Time')
            plt.ylabel('Magnitude')
            plt.title('Microlensing - Paczynski')
            plt.gca().invert_yaxis() 
            plt.show()