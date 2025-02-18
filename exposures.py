import matplotlib.pyplot as plt
from lsst.rsp import get_tap_service
import numpy as np
import imageio
import pandas as pd
import os
import gc
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from regions import CircleSkyRegion, PolygonSkyRegion

from matplotlib.patches import Circle
import time
import gc
import zipfile
import sys

from lsst.daf.butler import Butler, _dataset_ref
from lsst.daf.butler.registry import ConflictingDefinitionError
import lsst.daf.base as dafBase

import lsst.afw.display as afwDisplay
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.sphgeom
from shapely.geometry import Polygon, Point
from astropy.coordinates import SkyCoord
from spherical_geometry.polygon import SphericalPolygon

import astropy.units as u

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


from matplotlib import patches
from shapely.geometry import Point, Polygon
import lsst.afw.image as afwImage
from lsst.geom import SpherePoint, degrees,Point2D, Extent2I
from lsst.afw.display import Display
import warnings
warnings.filterwarnings("ignore", message="Spacing is not a multiple of base spacing")

class Calexp:
    def __init__(self, data_id):
        self.data_id = data_id
        if isinstance(data_id, afwImage._exposure.ExposureF):
            self.expF = self.data_id
        elif isinstance(data_id,_dataset_ref.DatasetRef):
            self.data_id = {'visit': dataRef.dataId['visit'], 'detector': dataRef.dataId['detector']}
            self.expF = butler.get('calexp', **self.data_id)
        elif isinstance(data_id, str):
            self.expF = afwImage.ExposureF(self.data_id)
        else:
            self.expF = butler.get('calexp', **data_id)
        self.wcs = self.expF.getWcs()
        self.cut = False
        self.warped  = False

    @property
    def center(self):
        return self.wcs.getSkyOrigin()

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

    def overlaps(self, region):
        ra_corners, dec_corners = self.get_corners()
        calexp_polygon = Polygon(zip(ra_corners, dec_corners))
        if isinstance(region, CircleSkyRegion):
            center = region.center
            radius_deg = region.radius.to(u.deg).value  # Convert to degrees
            circle = Point(center.ra.deg, center.dec.deg).buffer(radius_deg, resolution=50)
            return calexp_polygon.intersects(circle)
        
        elif isinstance(region, Polygon):
            return calexp_polygon.intersects(region)
        else:
            raise ValueError("Unsupported region type")


    def check_edge(self, ra, dec, d=50):
        """Check if a point is near the edge of the calexp."""
        x, y = self.sky_to_pix(ra, dec)
        bbox = self.expF.getBBox()
        return (
            abs(bbox.minX - x) < d or 
            abs(bbox.maxX - x) < d or 
            abs(bbox.minY - y) < d or 
            abs(bbox.maxY - y) < d)


    def cutout(self,roi):
        exp=self.expF.getCutout(Point2D(*self.sky_to_pix(*roi[0])), Extent2I(roi[1]))
        cutout = Calexp(self.data_id)
        cutout.wcs = exp.getWcs()
        cutout.expF = exp
        return cutout
    

    def plot(self, title=None, fig=None, ax=None, warp=None, roi=None, ticks=8, cut_size=401, col=None, figsize=None, n_ticks = (10,10), show=True):
        '''warp: calexp_ref'''
        return_ax = False
        if fig is None:
            fig = plt.figure(figsize=figsize)
        if ax is None and warp is None:
            ax = plt.subplot(projection=WCS(self.wcs.getFitsMetadata()))
            return_ax = True
        expF = self.expF
        if warp is not None:
            config = RegisterConfig()
            task = RegisterTask(name="register", config=config)
            expF = task.warpExposure(self.expF, self.wcs, warp.wcs, warp.expF.getBBox())
            ax = plt.subplot(projection=WCS(warp.wcs.getFitsMetadata()))
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
        # ra_corners, _ = self.get_corners()
        # space = max((abs(ra_corners[1] - ra_corners[0])) / (ticks + 2), 0.1)  # Minimum spacing of 0.1 degrees
        # if roi is not None:
        #     scale_factor = roi[1] / min(self.expF.getDimensions())
        #     space = max(space * scale_factor, 0.1)  # Ensure spacing is at least 0.1 degrees

        ax.coords['ra'].set_ticks(spacing=space * u.deg)  
        ax.coords['dec'].set_ticks(spacing=space * u.deg)

        ax.coords['ra'].set_major_formatter('dd:mm:ss')
        ax.coords['dec'].set_major_formatter('dd:mm:ss')
        ax.grid(color='white', ls='--', lw=0.2)
        if roi is not None:
            if warp is None:
                x,y = self.sky_to_pix(*roi[0])
            else:
                x,y = warp.sky_to_pix(*roi[0])
            size = roi[1]
            ax.set_xlim(x - size / 2, x + size / 2)
            ax.set_ylim(y - size / 2, y + size / 2)
        
        im = plt.imshow(expF.image.array, cmap='gray', vmin=-200.0, vmax=400,origin='lower')
        if return_ax:
            return ax
            
    def add_point(self, ax, ra, dec, r=5, c="r", label=None):
        x, y = self.sky_to_pix(ra,dec)
        circle = Circle((x, y), radius=r, edgecolor=c, facecolor="none")
        circle = Circle((x, y), radius=r, edgecolor=c, facecolor="none", label=label)
        ax.add_patch(circle)

    def save_plot(self, ax, image_path, show=False):
        ax.figure.savefig(image_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(ax.figure)
        gc.collect()

    def get_mag(self, flux, flux_err):
        return flux_to_mag(self.expF, flux, flux_err)

    def get_sources(self, detect_task, schema):
        tab = afwTable.SourceTable.make(schema)
        result = detect_task.run(tab, self)
        return result.sources


def flux_to_mag(exposure, flux, flux_err):
    """Convert flux to magnitude."""
    photoCalib = exposure.getPhotoCalib()
    measure = photoCalib.instFluxToMagnitude(flux, flux_err)
    return measure.value, measure.error