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
from io import StringIO


from lsst.daf.butler import Butler
from lsst.daf.butler.registry import ConflictingDefinitionError
import lsst.daf.base as dafBase

import lsst.afw.display as afwDisplay
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.sphgeom
from lsst.sphgeom import Region
from lsst.geom import SpherePoint, Angle, degrees

from lsst.source.injection import ingest_injection_catalog, generate_injection_catalog
from lsst.source.injection import VisitInjectConfig, VisitInjectTask

from lsst.pipe.tasks.registerImage import RegisterConfig, RegisterTask
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask

from lsst.meas.algorithms.detection import SourceDetectionTask
from lsst.meas.deblender import SourceDeblendTask
from lsst.meas.base import SingleFrameMeasurementTask, ForcedMeasurementTask

from lsst.daf.butler import Butler
# butler_config = 'dp02'
butler_config = 'dp02-direct'
collections = '2.2i/runs/DP0.2'
butler = Butler(butler_config, collections=collections)

class LightCurve:
    def __init__ (self, ra=None, dec=None, band=None, data=None, event_id=None, name=None, model = None, path = None, params = None):
        '''data = pd.DataFrame(columns=["detector", "visit", "mjd", "mag_sim", "flux", "flux_err", "mag", "mag_err"])'''

        if path:
            self._load_from_file(path)
            data = path
            self.path = path
            self._load_dataframe(data, columns=["detector", "visit", "mjd", "flux", "flux_err", "mag", "mag_err", "mag_inj"])
        else:
            self.ra = ra
            self.dec = dec
            self.event_id = event_id
            self.band = band
            self.model = model
            self.params = params
            self.calexp_data_ref = None
            self.calexp_dataIds = None
            self._load_dataframe(data, columns=["detector", "visit", "mjd", "flux", "flux_err", "mag", "mag_err", "mag_inj"])


    def __str__(self):
        return (f"LightCurve ({self.ra}, {self.dec}) - Band {self.band} - Event ID: {self.event_id} - Points: {len(self.data)}")
    def __repr__(self):
        return (f"LightCurve ({self.ra}, {self.dec}) - Band {self.band} - Event ID: {self.event_id} - Points: {len(self.data)}")

    def _load_dataframe(self, data, columns):
        if data is None:
            self.data = pd.DataFrame(columns=columns)
        elif isinstance(data, str): 
            self.data = pd.read_csv(data, comment="#")
            self.band = data.split(".")[-2][-1]
        else:
            self.data = data

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
            self.data["mag_inj"] = m_t
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

    def add_data(self, data_id, values, columns = ["flux", "flux_err", "mag", "mag_err"]):
        '''columns = ["flux", "flux_err", "mag", "mag_err"]'''
        self.data.loc[(self.data["visit"] == data_id["visit"]) &  (self.data["detector"] == data_id["detector"]), columns] = values        

    def calc_mag(self, flux, flux_err, dataId = None, exposure = None):
        '''If exposure is given, then values are expected to be fluxes and needs to be transformed to magnitude.'''
        if dataId != None:
            self.data.loc[(self.data["visit"] == dataId["visit"]) & (self.data["detector"] == dataId["detector"]), "flux"] = flux
            self.data.loc[(self.data["visit"] == dataId["visit"]) & (self.data["detector"] == dataId["detector"]), "flux_err"] = flux_err
        if exposure!=None:
            photoCalib = exposure.getPhotoCalib()
            measure = photoCalib.instFluxToMagnitude(value, value_err)
            mag = measure.value; mag_err = measure.error
        else:
            mag, mag_err = Calexp(dataId).get_mag(flux, flux_err)
        if dataId != None:
            self.data.loc[(self.data["visit"] == dataId["visit"]) & (self.data["detector"] == dataId["detector"]), "mag"] = value
            self.data.loc[(self.data["visit"] == dataId["visit"]) & (self.data["detector"] == dataId["detector"]), "mag_err"] = value_err
        if exposure!=None:
            return value, value_err
            # print(f"ra = {ra_deg}, dec = {dec_deg}")
            # print("Measured ", measure)
            # print("Injected ", lc.data["mag"][j])

    def plot(self, title=None, sliced="all", show=True, mag_lim=(31,14), simulated=True, figsize=(10,4)):
        if sliced == "all":
            df = self.data
        else:
            df = self.data[sliced]
        if len(df["mjd"].values)==0:
            print(f"No points to plot for event {self.event_id} on band {self.band}.")
        else:
            df = df.sort_values(by=['mjd'])
            label_sim = "Simulated "
            label_mea = "Measured "
            edge_color = "none"
            if self.band is not None:
                bands_colors = {'u': 'b', 'g': 'c', 'r': 'g', 'i': 'orange', 'z': 'r', 'y': 'm'}
                if self.band in bands_colors:
                    edge_color = bands_colors[self.band]
                label_sim += self.band
                label_mea += self.band
    
            # plt.plot(df['mjd'], df['mag_sim'], label=label_sim, color='none', marker='o', alpha=0.6, markeredgecolor=edge_color, linestyle='', markersize=4, mew=1)
            if simulated:
                if self.model == "Pacz":  # params = {t_0, t_E, u_0, m_base}
                
                    def Pacz(t, t_0, t_E, u_0, m_base):
                        u_t = np.sqrt(u_0**2 + ((t - t_0) / t_E)**2)
                        A_t = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))
                        return m_base - 2.5 * np.log10(A_t)
                x = np.linspace(min(df['mjd']), max(df['mjd']), 500)
                m_t = Pacz(x, **self.params)
                plt.plot(x, m_t, color=edge_color,  alpha=0.4, linestyle='-', label=label_sim)
            plt.errorbar(df['mjd'], df['mag'], yerr=df['mag_err'], label=label_mea, color=edge_color, linestyle='', marker='o', capsize=4, markersize=5) 
            
            if figsize is not None:
                plt.gcf().set_size_inches(*figsize)
            plt.gca().invert_yaxis() 
            if mag_lim is not None:
                plt.ylim(*mag_lim)
            plt.gca().invert_yaxis() 
            
            
            plt.xlabel('Epoch (MJD)')
            plt.ylabel('mag')
            plt.legend(loc=(1.02, 0.01)) 
        
            if title is None:
                title = str(self)
            plt.title(title)
        
            if show:
                plt.show()


    def _load_from_file(self, path):
        with open(path, "r") as f:
            metadata = {}
            data_lines = []
            for line in f:
                if line.startswith("#"): 
                    key, value = line[2:].strip().split(": ", 1)
                    metadata[key] = value
                else:
                    data_lines.append(line)

        self.ra = float(metadata.get("ra", "nan"))
        self.dec = float(metadata.get("dec", "nan"))
        self.event_id = path.split("/")[-1].split("_")[1]
        self.band = metadata.get("band")
        self.model = metadata.get("model")
        self.params = metadata.get("params")
        self.data = pd.read_csv(StringIO("".join(data_lines)))
        
    def save(self, path):
        with open(path, "w") as f:
            f.write(f"# ra: {self.ra}\n")
            f.write(f"# dec: {self.dec}\n")
            f.write(f"# event_id: {self.event_id}\n")
            f.write(f"# band: {self.band}\n")
            f.write(f"# model: {self.model}\n")
            f.write(f"# params: {self.params}\n")
            self.data.to_csv(f, index=False)