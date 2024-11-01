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


afwDisplay.setDefaultBackend('matplotlib')
plt.style.use('tableau-colorblind10')

butler_config = 'dp02'
collections = '2.2i/runs/DP0.2'
butler = Butler(butler_config, collections=collections)


from matplotlib import patches
from shapely.geometry import Point, Polygon
import lsst.afw.image as afwImage
from lsst.geom import SpherePoint, degrees,Point2D, Extent2I
from lsst.afw.display import Display
import warnings
warnings.filterwarnings("ignore", message="Spacing is not a multiple of base spacing")

class Calexp:
    def __init__(self, calexp_data):
        self.calexp_data = calexp_data
        self.expF = butler.get('calexp', **calexp_data)
        self.wcs = self.expF.getWcs()
        self.cut = False
        self.warped  = False

    def pix_to_sky(self, x, y):
        """Convert pixel coordinates to RA and Dec."""
        sphere_point = self.wcs.pixelToSky(geom.Point2D(x, y))
        return sphere_point.getRa().asDegrees(), sphere_point.getDec().asDegrees()

    def sky_to_pix(self, ra, dec):
        """Convert RA and Dec to pixel coordinates."""
        xy = self.wcs.skyToPixel(SpherePoint(ra * degrees, dec * degrees))
        return int(np.round(xy.getX())), int(np.round(xy.getY()))

    def get_corners(self, coord="sky"):
        x0 = float(self.expF.getX0())
        y0 = float(self.expF.getY0())
        width = self.expF.getWidth()
        height = self.expF.getHeight()
        xcorners = [x0, x0+width, x0+width, x0]
        ycorners = [y0, y0, y0+width, y0+width] 
        ra_corners = []
        dec_corners = []
        if coord == "sky":
            for i in range(len(xcorners)):
                ra, dec = self.pix_to_sky(xcorners[i], ycorners[i])
                ra_corners.append(ra)
                dec_corners.append(dec)
            return ra_corners, dec_corners
        else:
            return xcorners, ycorners
            
    def contains(self, ra, dec):
        ra_corners, dec_corners = self.get_corners()
        polygon = Polygon(zip(ra_corners, dec_corners))
        if isinstance(ra, (float, int)):
            point = Point(ra, dec)
            return polygon.contains(point)
        else:
            return [polygon.contains(Point(rai,deci)) for rai, deci in zip(ra,dec)]

    def check_edge(self, ra, dec, d=50):
        """Check if a point is near the edge of the calexp."""
        x, y = self.sky_to_pix(ra, dec)
        bbox = self.expF.getBBox()
    
        # Comprobar si x o y están cerca de los bordes del bounding box
        return (
            abs(bbox.minX - x) < d or 
            abs(bbox.maxX - x) < d or 
            abs(bbox.minY - y) < d or 
            abs(bbox.maxY - y) < d)


    def cutout(self,roi):
        exp=self.expF.getCutout(Point2D(*self.sky_to_pix(*roi[0])), Extent2I(roi[1]))
        cutout = Calexp(self.calexp_data)
        cutout.wcs = exp.getWcs()
        cutout.expF = exp
        return cutout
    

    def plot(self, title=None, fig=None, ax=None, warp=None, roi=None, ticks=8, cut_size=401, col=None, figsize=None, n_ticks = (10,10)):
        '''warp: calexp_ref'''
  
        if fig is None:
            fig = plt.figure(figsize=figsize)
        if ax is None and warp is None:
            ax = plt.subplot(projection=WCS(self.wcs.getFitsMetadata()))
            return_ax = True
        expF = self.expF
        if warp is not None:
            config = RegisterConfig()
            task = RegisterTask(name="register", config=config)
            expF = task.warpExposure(self.expF, self.wcs, calexp_ref.wcs, calexp_ref.expF.getBBox())
            ax = plt.subplot(projection=WCS(calexp_ref.wcs.getFitsMetadata()))
            return_ax = True
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('RA (degrees)', fontsize=8)
        ax.set_ylabel('Dec (degrees)', fontsize=8)
        ax.coords['ra'].set_format_unit(u.deg)

        # ax.coords['ra'].set_ticks(number=n_ticks[0])
        # ax.coords['dec'].set_ticks(number=n_ticks[1])
        ax.coords['ra'].set_ticklabel(rotation=30, fontsize=8, pad = 15)
        # ax.coords['dec'].set_ticklabel(rotation=-30, fontsize=6)

        ra_corners, _ = self.get_corners()
        space = (abs(ra_corners[1]-ra_corners[0]))/(ticks+2)
        if roi is not None:
            space*=(roi[1]/min(self.expF.getDimensions()))
        ax.coords['ra'].set_ticks(spacing=space * u.deg)  
        ax.coords['dec'].set_ticks(spacing=space * u.deg)

        ax.coords['ra'].set_major_formatter('dd:mm:ss')
        ax.coords['dec'].set_major_formatter('dd:mm:ss')
        ax.grid(color='white', ls='--', lw=0.2)
        if roi is not None:
            if warp is None:
                x,y = self.sky_to_pix(*roi[0])
            else:
                x,y = calexp_ref.sky_to_pix(*roi[0])
            size = roi[1]
            ax.set_xlim(x - size / 2, x + size / 2)
            ax.set_ylim(y - size / 2, y + size / 2)
        
        im = plt.imshow(expF.image.array, cmap='gray', vmin=-200.0, vmax=400,origin='lower')
        if return_ax:
            return ax
            
    def add_point(self, ax, ra, dec, r=5, c="r"):
        x, y = self.sky_to_pix(ra,dec)
        circle = Circle((x, y), radius=r, edgecolor=c, facecolor="none")  # Ajusta `radius` como desees
        ax.add_patch(circle)
        
    def inject(self, inject_data, inject_task, fits=None):
        """Inject sources into the calexp."""
        try:
            injected_output = inject_task.run(
                injection_catalogs=[inject_data],
                input_exposure=self.calexp.clone()
            )
            injected_exposure = injected_output.output_exposure
            if fits is not False:
                injected_exposure.writeFits(fits)
            return (self.calexp_data.dataId, inject_data['exp_midpoint'], inject_data['mag'], injected_exposure)
        except Exception as e:
            print('No sources to inject for visit ', inject_data['visit'], "Error:", e)
            return None

    def to_mag(self, flux, flux_err):
        """Convert flux to magnitude."""
        photoCalib = self.calexp.getPhotoCalib()
        measure = photoCalib.instFluxToMagnitude(flux, flux_err)
        return measure.value, measure.error