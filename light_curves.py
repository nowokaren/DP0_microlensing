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
# import times
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

# butler_config = 'dp02'
butler_config = 'dp02-direct'
collections = '2.2i/runs/DP0.2'
butler = Butler(butler_config, collections=collections)

class LightCurve:
    def __init__ (self, ra=None, dec=None, band="i", data=None, name=None, model = None, params = None):
        '''data = pd.DataFrame(columns=["detector", "visit", "mjd", "mag_sim", "flux", "flux_err", "mag", "mag_err"])'''
        self.ra = ra
        self.dec = dec
        self.htm_id = None
        self.band = band
        if data is None:
            self.data = pd.DataFrame(columns=["detector", "visit", "mjd", "mag_sim", "flux", "flux_err", "mag", "mag_err"])
        else:
            self.data = data
        self.model = model
        self.params = params
        self.calexp_data_ref = None
        self.calexp_dataIds = None

    def __str__(self):
        return (f"LightCurve ({self.ra}, {self.dec}) - Band {self.band} - Points: {len(self.data)}")
    def __repr__(self):
        return (f"LightCurve ({self.ra}, {self.dec}) - Band {self.band} - Points: {len(self.data)}")

    def calculate_htm_id(self, level=20):
        pixelization = lsst.sphgeom.HtmPixelization(level)
        htm_id = pixelization.index(
            lsst.sphgeom.UnitVector3d(
                lsst.sphgeom.LonLat.fromDegrees(self.ra, self.dec)))
        circle = pixelization.triangle(htm_id).getBoundingCircle()
        scale = circle.getOpeningAngle().asDegrees()*3600
        level = pixelization.getLevel()
        print(f'(ra,dec) = ({self.ra}, {self.dec})/nHTM_ID = {htm_id} - HTM_level={level} (bounded by a circle of radius ~{scale:0.2f} arcsec.)')
        self.htm_id = htm_id
        return htm_id

    def collect_calexp(self, level=20):
        if not isinstance(self.htm_id, int):
            self.calculate_htm_id(level)
        datasetRefs = list(butler.registry.queryDatasets(
            "calexp", htm20=self.htm_id, where=f"band = '{self.band}'"))
        print("{:<20}".format("") + f"Found {len(datasetRefs)} calexps")
        ccd_visit = butler.get('ccdVisitTable')
        mjds = []
        detectors = [] ; visits = []
        nans = []


        for calexp_data in datasetRefs:
            did = calexp_data.dataId
            ccdrow = (ccd_visit['visitId'] == did['visit']) & (
                ccd_visit['detector'] == did['detector'])
            exp_midpoint = ccd_visit[ccdrow]['expMidptMJD'].values[0]
            mjds.append(exp_midpoint)
            detectors.append(did['detector'])  
            visits.append(did['visit'])          
            nans.append(np.nan)
            
        new_data = pd.DataFrame({
            "detector": detectors,
            "visit": visits,
            "mjd": mjds,
            "mag_sim": nans,
            "flux": nans,
            "flux_err": nans,
            "mag": nans,
            "mag_err": nans})

        # Eliminar filas que son completamente NaN
        new_data = new_data.dropna(how='all')
        
        # Verificar que new_data no esté vacío antes de concatenar
        if not new_data.empty:
            self.data = pd.concat([self.data, new_data], ignore_index=True)
        self.calexp_data_ref = datasetRefs
        self.calexp_dataIds = [{"visit": dataid.dataId["visit"], "detector": dataid.dataId["detector"]} for dataid in datasetRefs]

    def simulate(self, params, model="Pacz", plot=False):
        """
        Simulate magnitudes for a microlensing event and optionally plot the result.
    
        Parameters:
        params (dict): Dictionary containing parameters like t_0, t_E, u_0, and m_base.
        model (str): The model to use for simulation. Default is "Pacz".
        plot (bool): If True, plot the magnification curve.
        
        Returns:
        np.ndarray: Magnitudes corresponding to the mjds in self.data["mjd"].
        """
    
        if model == "Pacz":  # params = {t_0, t_E, u_0, m_base}
            
            def Pacz(t, t_0, t_E, u_0, m_base):
                u_t = np.sqrt(u_0**2 + ((t - t_0) / t_E)**2)
                A_t = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))
                return m_base - 2.5 * np.log10(A_t)
            m_t = Pacz(self.data["mjd"], **params)
            self.data["mag_sim"] = m_t
            self.name = "Simulated"
            self.model = "Pacz"
        else:
            raise ValueError("Model not recognized. Currently supported models: 'Pacz'.")
        self.params = params
        if plot:
            t_plot = np.linspace(np.min(self.data["mjd"]), np.max(self.data["mjd"]), 1000)
            m_plot = Pacz(t_plot, **params)
            plt.plot(t_plot, m_plot, color='gray')
            plt.scatter(self.data["mjd"], m_t, color='red')
            plt.xlabel('mjd')
            plt.ylabel('Magnitude')
            plt.title('Microlensing - Paczynski')
            plt.gca().invert_yaxis() 
            plt.show()
        

    def add_flux(self, flux, flux_err, dataId):
        print(self.data.loc[(self.data["visit"] == dataId["visit"]) & (self.data["detector"] == dataId["detector"]), "flux"])
        self.data.loc[(self.data["visit"] == dataId["visit"]) & (self.data["detector"] == dataId["detector"]), "flux"] = flux
        self.data.loc[(self.data["visit"] == dataId["visit"]) & (self.data["detector"] == dataId["detector"]), "flux_err"] = flux_err



    def add_mag(self, value, value_err, dataId, exposure = None):
        '''If exposure is given, then values are expected to be fluxes and needs to be transformed to magnitude.'''
        if exposure!=None:
            photoCalib = exposure.getPhotoCalib()
            measure = photoCalib.instFluxToMagnitude(value, value_err)
            value = measure.value; value_err = measure.error
        self.data.loc[(self.data["visit"] == dataId["visit"]) & (self.data["detector"] == dataId["detector"]), "mag"] = value
        self.data.loc[(self.data["visit"] == dataId["visit"]) & (self.data["detector"] == dataId["detector"]), "mag_err"] = value_err
        if exposure!=None:
            return value, value_err
            # print(f"ra = {ra_deg}, dec = {dec_deg}")
            # print("Measured ", measure)
            # print("Injected ", lc.data["mag"][j])

    def plot(self, sliced = "all"):
        if sliced == "all":
            df = self.data
        else:
            df = self.data[sliced]
        plt.figure(figsize=(10, 6))
        plt.plot(df['mjd'], df['mag_sim'], label='Magnitud Simulada', color='gray', marker='o')
        plt.errorbar(df['mjd'], df['mag'], yerr=df['mag_err'], label='Magnitud Medida', color='red', linestyle='', marker='.', capsize=3)
        plt.xlabel('MJD (Modified Julian Date)')
        plt.ylabel('Magnitud')
        plt.title(str(self))
        plt.gca().invert_yaxis()  
        plt.legend()
        plt.show()