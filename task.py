dp0_limits = [[48, 76], [-44,-28]] # [ra_lim, dec_lim]
import random
import time
from tqdm.notebook import tqdm
import os
from datetime import datetime
import numpy as np
import pandas as pd
from astropy.table import Table, vstack
from lsst.afw import table as afwTable
from lsst.meas.base import SingleFrameMeasurementTask
from lsst.meas.algorithms import SourceDetectionTask
from lsst.source.injection import (
    ingest_injection_catalog, generate_injection_catalog,
    VisitInjectConfig, VisitInjectTask
)
import traceback
from spherical_geometry.polygon import SphericalPolygon

import lsst.daf.base as dafBase
from light_curves import LightCurve
from exposures import Calexp
from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from astropy.coordinates import SkyCoord
from lsst.sphgeom import HtmPixelization, UnitVector3d, LonLat
from lsst.geom import Angle, radians, degrees, SpherePoint
from shapely.geometry import Polygon, Point
from astropy.coordinates import SkyCoord
import astropy.units as u
from regions import CircleSkyRegion, PolygonSkyRegion
from itertools import product
from astropy.table import Table
from lsst.meas.base import ForcedMeasurementTask

from tools import tri_sample, triangle_min_height, circ_sample
from lsst.sphgeom import Region
import re

from lsst.daf.butler import Butler
# butler_config = 'dp02'
butler_config = 'dp02-direct'
collections = '2.2i/runs/DP0.2'
butler = Butler(butler_config, collections=collections)


class Run:
#     def __init__(self, ra, dec, scale, bands, density=1000, name=None, main_path="runs/", data_events=None, data_calexp=None, calexp_method="htm", measure_method="ForcedMeas"):
#         self.density = density                                                   # Sources injected per deg²
#         self.ra = ra ; self.dec = dec                                            # Center of the area to be injected
#         self.calexps_method = calexp_method                                      # Method used to collect calexps
#         self.measure_method = measure_method
#         self.bands = bands                                                       # Bands or filter to inject
#         self.inj_lc = []                                                         # List of LightCurves to be injected
#         date = datetime.now()
#         self.name = name if name else date.strftime("%Y%m%d_%H%M%S")             # Name of the Run
#         self.main_path = os.path.join(main_path, self.name)                      # Path to save the results
#         os.makedirs(self.main_path, exist_ok=True)
#         self.tasks = {}                                                          # Save tasks to use
#         self.log = {"task": ["Start"], "time": [time.time()], "detail": [None]}  # Log of tasks and their time consumption
        
#         if data_events is None:                                                  # DataFrame with data of events to be injected
#             self.data_events = pd.DataFrame(columns=["event_id", "ra", "dec", "model", "band", "points"])
#         elif isinstance(data_events, str):
#             self.data_events = pd.read_csv(data_events)
#         else:
#             self.data_events = data_events
        
#         if data_calexp is None:                                                   # DataFrame with data of calexps to be injected
#             self.data_calexp = pd.DataFrame(columns=["detector", "visit", "mjd", "band", "overlap", "ids_events"])
#         elif isinstance(data_calexp, str):
#             self.data_calexp = pd.read_csv(data_calexp)
#         else:
#             self.data_calexp = data_calexp

#         self.datasetRefs = None                                                   # Objects of DataRef of calexps to be injected
#         self.dist = None                                                          # Minimum distance between injected sources
#         self.ref_dist = None                                                      # Order of magnitude of the injected area
#         self.inject_table = {band: None for band in self.bands}                   # Sources to be injected (input format of Injection Task)
        
#         if self.calexps_method == "htm":
#             self.htm_level = scale  # Level of the HTM triangle to be injected
#             pixelization = HtmPixelization(self.htm_level)
#             self.htm_id = pixelization.index(UnitVector3d(LonLat.fromDegrees(self.ra, self.dec)))  # ID of the HTM triangle to be injected
#             self.bound_circle_radius = pixelization.triangle(self.htm_id).getBoundingCircle().getOpeningAngle().asDegrees() * 3600
#             htm_triangle = pixelization.triangle(self.htm_id)
#             tri_ra_dec = []
#             for vertex in htm_triangle.getVertices():
#                 lon = LonLat.longitudeOf(vertex).asDegrees()
#                 lat = LonLat.latitudeOf(vertex).asDegrees()
#                 tri_ra_dec.append((lon, lat))
#             self.htm_vertex = tri_ra_dec  # Vertices of the HTM triangle to be injected
#             self.level_area = np.pi / (2 * 4**(self.htm_level - 1)) * (180 / np.pi)**2  # Area of the sky to be injected (deg²)
#             ra_vertices, dec_vertices = zip(*tri_ra_dec)
#             self.region = SphericalPolygon.from_lonlat(ra_vertices, dec_vertices, degrees=True)

        
#         elif self.calexps_method == "overlap":
#             self.radius = scale  # Radius of the area to be injected
#             center = SkyCoord(ra=self.ra, dec=self.dec, unit="deg", frame="icrs")
#             self.region = CircleSkyRegion(center=center, radius=self.radius * u.deg)
#             self.area =  np.pi * self.region.radius.to(u.deg).value ** 2  # Area in square degrees

#         else:
#             raise ValueError("Unknown method. Choose either 'htm' or 'overlap'.")
#         self.n_lc = int(self.density * self.area) + 1  # Number of sources to be injected
#         self.write_log()
    def __init__(self, name=None, ra=None, dec=None, bands=None, calexps_method=None, 
                 measure_method=None, density=None, area=None, radius=None, n_lc=None, 
                 main_path="./runs", data_events=None, data_calexp=None, scale=None):

        self.name = name if name else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_path = os.path.join(main_path, self.name)
        os.makedirs(self.main_path, exist_ok=True)  
        from_path = all(arg is None for arg in [ra, dec, bands, calexps_method, measure_method, density, area, radius, n_lc, data_events, data_calexp, scale])

        if from_path:
            log_path = os.path.join(self.main_path, f"{self.name}_log.txt")
            os.path.exists(log_path)
            print(f"Loading data from {log_path}")
            self._load_from_path(log_path)
            data_events = os.path.join(self.main_path, "data_events.csv")
            data_calexp = os.path.join(self.main_path, "data_calexp.csv")
        else:
            self.ra = ra
            self.dec = dec
            self.bands = bands if bands else []
            self.calexps_method = calexps_method
            self.measure_method = measure_method
            self.density = density
            self.area = area
            self.scale = scale
            self.n_lc = n_lc

        self.inj_lc = []  
        self.tasks = {}  
        self.log = {"task": ["Start"], "time": [time.time()], "detail": [None]}  
        
        self.data_events = self._load_dataframe(data_events, columns=["event_id", "ra", "dec", "model", "band", "points"])
        self.data_calexp = self._load_dataframe(data_calexp, columns=["detector", "visit", "mjd", "band", "overlap", "ids_events"])
        
        self.datasetRefs = None
        self.dist = None
        self.ref_dist = None
        self.inject_table = {band: None for band in self.bands}

        if self.calexps_method == "htm":
            self._setup_htm(self.scale)
        elif self.calexps_method == "overlap":
            self._setup_overlap(self.scale)
        else:
            raise ValueError("Unknown method. Choose either 'htm' or 'overlap'.")

        if self.density is not None and self.area is not None:
            self.n_lc = int(self.density * self.area) + 1  
        if ~from_path:
            self.write_log()

    def _load_dataframe(self, data, columns):
        if data is None:
            return pd.DataFrame(columns=columns)
        elif isinstance(data, str):
            return pd.read_csv(data)
        else:
            return data

    def _setup_htm(self, scale):
        self.htm_level = scale
        pixelization = HtmPixelization(self.htm_level)
        self.htm_id = pixelization.index(UnitVector3d(LonLat.fromDegrees(self.ra, self.dec)))
        self.bound_circle_radius = pixelization.triangle(self.htm_id).getBoundingCircle().getOpeningAngle().asDegrees() * 3600
        htm_triangle = pixelization.triangle(self.htm_id)

        tri_ra_dec = [(LonLat.longitudeOf(v).asDegrees(), LonLat.latitudeOf(v).asDegrees()) for v in htm_triangle.getVertices()]
        self.htm_vertex = tri_ra_dec
        self.level_area = np.pi / (2 * 4**(self.htm_level - 1)) * (180 / np.pi)**2  # Area in deg²

        ra_vertices, dec_vertices = zip(*tri_ra_dec)
        self.region = SphericalPolygon.from_lonlat(ra_vertices, dec_vertices, degrees=True)

    def _setup_overlap(self, scale):
        self.radius = scale
        center = SkyCoord(ra=self.ra, dec=self.dec, unit="deg", frame="icrs")
        self.region = CircleSkyRegion(center=center, radius=self.radius * u.deg)
        self.area = np.pi * self.region.radius.to(u.deg).value ** 2  # Área en deg²

    def _load_from_path(self, log_path):
        with open(log_path, "r") as file:
            log_content = file.readlines()

        patterns = {
            "ra": r"Center: RA=([\d\.\-]+),",
            "dec": r"Dec=([\d\.\-]+)",
            "bands": r"Bands:\s*(.+)",
            "calexps_method": r"Calexp Method:\s*(.+)",
            "measure_method": r"Measure Method:\s*(.+)",
            "density": r"Density:\s*([\d\.]+)",
            "area": r"Area:\s*([\d\.]+)",
            "scale": r"Scale:\s*([\d\.]+)",
            "n_lc": r"Number of LightCurves to inject:\s*(\d+)"
        }

        for line in log_content:
            for key, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    setattr(self, key, match.group(1))

        self.ra = float(self.ra) if hasattr(self, "ra") else None
        self.dec = float(self.dec) if hasattr(self, "dec") else None
        self.density = float(self.density) if hasattr(self, "density") else None
        self.area = float(self.area) if hasattr(self, "area") else None
        self.scale = float(self.scale) if hasattr(self, "scale") else None
        self.n_lc = int(self.n_lc) if hasattr(self, "n_lc") else None
        self.bands = self.bands if hasattr(self, "bands") else None

    def write_log(self):
        """Guarda la configuración actual en un archivo de log."""
        separator = "=" * 50
        log_content = (
            f"{separator}\n"
            f"Run Name: {self.name}\n"
            f"Center: RA={self.ra}, Dec={self.dec}\n"
            f"Bands: {self.bands}\n"
            f"Calexp Method: {self.calexps_method}\n"
            f"Measure Method: {self.measure_method}\n"
            f"Density: {self.density} sources/deg²\n"
            f"Area: {self.area} deg²\n"
            f"Scale: {self.scale} \n"
            f"Number of LightCurves to inject: {self.n_lc}\n"
            f"Main Path: {self.main_path}\n"
            f"{separator}"
        )
        log_path = os.path.join(self.main_path, f"{self.name}_log.txt")
        with open(log_path, "w") as log_file:
            log_file.write(log_content)

        print(f"Log saved in: {log_path}")

    def __str__(self):
        separator = "-" * 40 
        return (f"{separator}\n"
                f"Run Name: {self.name}\n"
                f"Center: RA={self.ra}, Dec={self.dec}\n"
                f"Band: {self.bands}\n"
                f"Calexp Method: {self.calexps_method}\n"
                f"Measure Method: {self.measure_method}\n"
                f"Density: {self.density} sources/deg²\n"
                f"Area: {self.area:.3f} deg²\n"
                f"Scale: {self.radius:.3f} deg\n"
                f"Number of LightCurves to inject: {self.n_lc}\n"
                f"Main Path: {self.main_path}\n"
                f"{separator}")


    def __repr__(self):
        separator = "-" * 80 
        return (f"{separator}\n"
                f"Run Name: {self.name}\n"
                f"Center: RA={self.ra}, Dec={self.dec}\n"
                f"Band: {self.bands}\n"
                f"Calexp Method: {self.calexps_method}\n"
                f"Measure Method: {self.measure_method}\n"
                f"Density: {self.density} sources/deg²\n"
                f"Area: {self.area:.3f} deg²\n"
                f"Scale: {self.radius:.3f} deg\n"
                f"Number of LightCurves to inject: {self.n_lc}\n"
                f"Main Path: {self.main_path}\n"
                f"{separator}")


    def mjd(self, dataId):
        return self.data_calexp.loc[(self.data_calexp["visit"] == dataId["visit"]) & (self.data_calexp["detector"] == dataId["detector"]), "mjd"].values[0]
        
    def log_task(self, name, det=None):
        self.log["time"].append(time.time())
        self.log["task"].append(name)
        self.log["detail"].append(det)

    def create_schema(self):
        schema = afwTable.SourceTable.makeMinimalSchema()
        schema.addField("coord_raErr", type="F", doc="Error in RA coordinate")
        schema.addField("coord_decErr", type="F", doc="Error in Dec coordinate")
        return schema

    def collect_calexp(self, n_max=None):
        bands_str = f"({', '.join(map(repr, self.bands))})"
        print("Collecting calexps...")
        if self.calexps_method == "htm":
            print(f'(ra,dec) = ({self.ra}, {self.dec}) \nHTM_ID = {self.htm_id} - HTM_level={self.htm_level} (bounded by a circle of radius ~{bound_circle_radius:0.2f} arcsec.)')
            self.datasetRefs = list(butler.registry.queryDatasets("calexp", htm20=self.htm_id, where=f"band IN {bands_str}"))
        elif self.calexps_method == "overlap":
            target_point = SpherePoint(Angle(self.ra, degrees), Angle(self.dec, degrees))
            RA = target_point.getLongitude().asDegrees()
            DEC = target_point.getLatitude().asDegrees()
            circle = Region.from_ivoa_pos(f"CIRCLE {RA} {DEC} {self.radius}")
            self.datasetRefs = butler.query_datasets("calexp", where=f"visit_detector_region.region OVERLAPS my_region AND band IN {bands_str}", bind={"ra": RA, "dec": DEC, "my_region": circle})
            print(f'(ra,dec) = ({self.ra}, {self.dec}) \nCircle of radius ~{self.radius:0.3f} deg.')

        if n_max is not None:
            datasetRefs = self.datasetRefs[:n_max]
        n_dataref = len(datasetRefs)
        print(f"Found {n_dataref} calexps.")
        ccd_visit = butler.get('ccdVisitTable')
        self.data_calexp["detector"] = [calexp_data.dataId['detector'] for calexp_data in datasetRefs]
        self.data_calexp["visit"] = [calexp_data.dataId['visit'] for calexp_data in datasetRefs]
        self.data_calexp["mjd"] = [
            ccd_visit[(ccd_visit['visitId'] == calexp_data.dataId['visit']) & 
                      (ccd_visit['detector'] == calexp_data.dataId['detector'])]['expMidptMJD'].values[0]
            for calexp_data in tqdm(datasetRefs, desc="Processing MJD values")
        ]
        self.data_calexp["band"] = [calexp_data.dataId['band'] for calexp_data in datasetRefs]
        self.log_task("Collecting calexps", det=n_dataref)

    def generate_location(self, dist=None):
        if dist is None:
            if self.calexps_method == "htm":
                self.ref_dist = triangle_min_height(self.htm_vertex)
            elif self.calexps_method == "overlap":
                self.ref_dist = self.radius
            self.dist = self.ref_dist / 20
        else:
            self.dist = dist
    
        distance = self.dist / 2
        if self.calexps_method == "htm":
            ra, dec = tri_sample(self.htm_vertex)
        elif self.calexps_method == "overlap":
            ra, dec = circ_sample(self.ra, self.dec, self.radius, margin=self.dist)
    
        if len(self.inj_lc) != 0:
            for lc in self.inj_lc:
                distance = np.sqrt((lc.ra - ra) ** 2 + (lc.dec - dec) ** 2)
                if distance < self.dist: 
                    return self.generate_location(dist=self.dist) 

        return ra, dec

    def add_lc(self, ra, dec, params, event_id, band, model="Pacz", plot=False):
        lc = LightCurve(ra, dec, band=band)
        lc.data["mjd"] = self.data_calexp[self.data_calexp["band"]==band]["mjd"]
        lc.data["visit"] = self.data_calexp[self.data_calexp["band"]==band]["visit"]
        lc.data["detector"] = self.data_calexp[self.data_calexp["band"]==band]["detector"]
        lc.simulate(params, model=model, plot=plot)
        lc.event_id = event_id
        new_lc = {"event_id":event_id, "ra": ra, "dec": dec, "model": model, "band": band}
        for key in params.keys():
            if key not in self.data_events.columns:
                self.data_events[key] = None  
            new_lc[key] = params[key] 
        self.data_events.loc[len(self.data_events)] = new_lc
        self.inj_lc.append(lc)

    def inject_task(self):
        inject_config = VisitInjectConfig()
        self.tasks["Injection"] = VisitInjectTask(config=inject_config)

    
    def create_injection_table(self, calexp, band):
        catalog = []
        visit = calexp.data_id["visit"]
        detector = calexp.data_id["detector"]
        mjd = self.data_calexp[(self.data_calexp["visit"] == visit) & (self.data_calexp["detector"] == detector)]["mjd"].values[0]
        for lc in self.inj_lc:
            if lc.band == band:
                lc_calexp = lc.data[(lc.data["visit"] == visit) & (lc.data["detector"] == detector)]
                catalog.append([visit, detector, lc.ra, lc.dec, "Star", mjd, lc_calexp["mag_inj"].values[0]])    
        return Table(rows=catalog,names=["visit", "detector", "ra", "dec", "source_type", "exp_midpoint", "mag"])

    def check_injection_catalog(self, calexp, catalog, before_injection = True):
        ra, dec = catalog["ra"], catalog["dec"]
        if before_injection:
            mask_visit = np.array(catalog["visit"] == calexp.data_id["visit"])
            mask_detector = np.array(catalog["detector"] == calexp.data_id["detector"])
            mask_contain = np.array(calexp.contains(ra, dec)) 
            if False in mask_contain:
                print("Light curves NOT contained: ", len([i for i in range(len(ra)) if not mask_contain[i]]))
            mask_edge = np.array([calexp.check_edge(r, d, d=100) for r, d in zip(ra, dec)])
            if True in mask_edge:
                print("Light curves near edge: ", len([i for i in range(len(ra)) if mask_edge[i]]))
            keep_mask = mask_contain & ~mask_edge & mask_detector & mask_visit
        else:
            mask_flag = np.array([i!=0 for i in catalog["injection_flag"]])
            if True in mask_flag:
                print("Light curves marked FLAG: ", [i for i in range(len(ra)) if mask_flag[i]])
            keep_mask = ~mask_flag
        filtered_catalog = catalog[keep_mask]
        data_id = calexp.data_id
        self.data_calexp.loc[(self.data_calexp["detector"] == data_id["detector"]) & (self.data_calexp["visit"] == data_id["visit"]), "ids_events"] = "-".join(map(str, self.data_events[self.data_events["ra"].isin(filtered_catalog["ra"].value)].index))
        return filtered_catalog
            
    def inject_calexp(self, calexp, inject_table, save_fit = None):
        '''Creates injecting catalog and inject light curve's points if the calexp contains it.
        save_fit = name of the file to be saved'''

        exposure = calexp.expF
        try:
            injected_output = self.tasks["Injection"].run(
                injection_catalogs=[inject_table],
                input_exposure=exposure.clone(),
                psf=exposure.getPsf(),
                photo_calib=exposure.getPhotoCalib(),
                wcs=calexp.wcs)
        except Exception as e:
            print("Couldn't inject the injection_catalog")
            print("Error:", str(e))
            print("Detailed traceback:")
            print(traceback.format_exc())  # This will print the full traceback
            return None, None

        injected_exposure = injected_output.output_exposure
        injected_catalog = injected_output.output_catalog
        self.log_task("Injection", det = len(injected_catalog))
        injected_exposure.writeFits(self.main_path+"/"+save_fit)
        return injected_exposure, injected_catalog

########### MEASUREMENT methods ################

    def measure_task(self):
        schema = afwTable.SourceTable.makeMinimalSchema()
        if self.measure_method == "SingleFrame":
            raerr = schema.addField("coord_raErr", type="F")
            decerr = schema.addField("coord_decErr", type="F")
            algMetadata = dafBase.PropertyList()
            config = SourceDetectionTask.ConfigClass()
            config.thresholdValue = 5
            config.thresholdType = "stdev"
            self.tasks["Detection"] = SourceDetectionTask(schema=schema, config=config)
            config = SingleFrameMeasurementTask.ConfigClass()
            self.tasks["Measurement"] = SingleFrameMeasurementTask(schema=schema,
                                                               config=config,
                                                               algMetadata=algMetadata)

        elif self.measure_method  == "ForcedMeas":
            alias = schema.getAliasMap() 
            x_key = schema.addField("centroid_x", type="D")
            y_key = schema.addField("centroid_y", type="D")
            alias.set("slot_Centroid", "centroid")
            
            xx_key = schema.addField("shape_xx", type="D")
            yy_key = schema.addField("shape_yy", type="D")
            xy_key = schema.addField("shape_xy", type="D")
            alias.set("slot_Shape", "shape")
            type_key = schema.addField("type_flag", type="F")
            config = ForcedMeasurementTask.ConfigClass()
            config.copyColumns = {}
            config.plugins.names = [
                "base_TransformedCentroid",
                "base_PsfFlux",
                "base_TransformedShape"
            ]
            config.doReplaceWithNoise = False
            self.tasks["Measurement"]  = ForcedMeasurementTask(schema, config=config)
        return schema

    # def create_sources_table(self, pre_sources):
    #     '''- self.measure_method  = "ForcedMeas" -> pre_sources = injected_catalog
    #        - self.measure_method  = "SingleFrame" -> pre_sources = calexp"'''
    #     if self.measure_method  == "ForcedMeas":
    #         sources = afwTable.SourceCatalog(schema)
    #         for source in pre_sources:
    #             sourceRec = sources.addNew()
    #             coord = geom.SpherePoint(geom.Angle(source["ra"], geom.degrees).asRadians(),geom.Angle(source["dec"], geom.degrees).asRadians(), geom.radians)
    #             sourceRec.setCoord(coord)
    #             sourceRec["centroid_x"], sourceRec["centroid_y"]= new_calexp.sky_to_pix(source["ra"], source["dec"])
    #             sourceRec["type_flat"] = 0
    #         self.log_task("Creating table to measure", det=len(sources))
    #     elif self.measure_method == "SingleFrame":
    #         tab = afwTable.SourceTable.make(schema)
    #         result = self.tasks["Detection"].run(tab, pre_sources)
    #         sources = result.sources
    #         self.log_task("Detection", det=len(sources))
    #     return sources
            

    # def measure_calexp(self, exposure, sources, schema):
    #     if self.measure_method  == "SingleFrame":
    #         self.tasks["Measurement"].run(measCat=sources, exposure=exposure)
    #     elif self.measure_method  == "ForcedMeas":
    #         forcedMeasCat = self.tasks["Measurement"].generateMeasCat(exposure, sources, exposure.getWcs())
    #         self.tasks["Measurement"].run(forcedMeasCat, exposure, sources, exposure.getWcs())
    #         sources = forcedMeasCat.asAstropy()            
    #     self.log_task("Measurement", det = method)
    #     return sources


    # def find_flux(self, sources, ra, dec, save=None):
    #     distances = [SpherePoint(ra,dec, degrees).separation(SpherePoint(sources["coord_ra"][i],sources["coord_dec"][i], radians)) for i in range(len(sources))]
    #     id_near = np.argmin(distances)
    #     dist = distances[id_near]
    #     if dist>Angle(1e-6, radians):
    #         print(f"Source not found. Distance = {dist} ")
    #         return None, None
    #     return sources["base_PsfFlux_instFlux"][id_near], sources["base_PsfFlux_instFluxErr"][id_near]

    # def get_fluxes(self, sources):
    #     if self.measure_method == "SinlgeFrame":
    #         return 


    def measurement(self, schema, calexp, injected_catalog):
        table = pd.DataFrame(columns=["ra", "dec", "flux", "flux_err", "mag", "mag_err", "flag"])
        table["ra"] = injected_catalog["ra"]; table["dec"] = injected_catalog["dec"]
        
        if self.measure_method  == "ForcedMeas":
            sources = afwTable.SourceCatalog(schema)
            for source in injected_catalog:
                sourceRec = sources.addNew()
                coord = SpherePoint(Angle(source["ra"], degrees).asRadians(),
                                         Angle(source["dec"], degrees).asRadians(), radians)
                sourceRec.setCoord(coord)
                sourceRec["centroid_x"], sourceRec["centroid_y"]= calexp.sky_to_pix(source["ra"], source["dec"])
                sourceRec["type_flag"] = 0
            self.log_task("Creating table to measure", det=len(sources))
            forcedMeasCat = self.tasks["Measurement"].generateMeasCat(calexp.expF, sources, calexp.wcs)
            self.tasks["Measurement"].run(forcedMeasCat, calexp.expF, sources, calexp.wcs)
            sources = forcedMeasCat.asAstropy()
            for i, source in enumerate(sources):
                ra, dec = calexp.pix_to_sky(source["slot_Centroid_x"],source["slot_Centroid_y"])
                flux, flux_err = source["base_PsfFlux_instFlux"], source["base_PsfFlux_instFluxErr"]
                mag, mag_err = calexp.get_mag(flux, flux_err)
                table.loc[i, ["flux", "flux_err", "mag", "mag_err", "flag"]] = [flux, flux_err, mag, mag_err, 0]
            self.log_task("Measurement", det = len(table))
        elif self.measure_method == "SingleFrame":
            tab = afwTable.SourceTable.make(schema)
            # DETECTION
            result = self.tasks["Detection"].run(tab, calexp.expF)
            sources = result.sources
            self.log_task("Detection", det=len(sources))
            # MEASURE ALL DETECTED SOURCES
            self.tasks["Measurement"].run(measCat=sources, exposure=calexp.expF)
            self.log_task("Measurement", det =  len(sources))
            # FIND INJECTED SOURCES IN MEASURED CATALOG
            for source in injected_catalog:
                flag = 0
                ra, dec = source["ra"],source["dec"]
                distances = [SpherePoint(ra, dec, degrees).separation(SpherePoint(sources["coord_ra"][i],sources["coord_dec"][i], radians)) for i in range(len(sources))]
                id_near = np.argmin(distances)
                dist = distances[id_near]
                if dist>Angle(1e-6, radians):
                    flag = dist
                flux, flux_err = sources["base_PsfFlux_instFlux"][id_near], sources["base_PsfFlux_instFluxErr"][id_near]
                mag, mag_err = calexp.get_mag(flux, flux_err)
                table.loc[(table["ra"] == ra) & (table["dec"] == dec), ["flux", "flux_err", "mag", "mag_err", "flag"]] = [flux, flux_err, mag, mag_err, flag]
        return table, sources

    def sky_map(self, color='red', band=None, lwT=1, lwC=1, calexps=None, inj_points=True):
        if band ==None:
            inj_lc_list = self.inj_lc
        else:
            inj_lc_list = [lc for lc in self.inj_lc if lc.band==band]
        ra_vals = [lc.ra for lc in inj_lc_list]
        dec_vals = [lc.dec for lc in inj_lc_list]
        inj_points = [lc.data["mag"].count() for lc in inj_lc_list] 
        if self.calexps_method == "htm":
            label = f"HTM level {self.htm_level}"
            title = f"Injected sources distribution in the HTM triangle (Level {self.htm_level})"
            region_polygon = Polygon(self.htm_vertex)
        elif self.calexps_method == "overlap":
            label = f"circle of radius {self.radius}"
            title = f"Injected sources distribution in the circle of radius {self.radius}"
            center = Point(self.ra, self.dec)
            region_polygon = center.buffer(self.radius) 
        if band!= None:
            title += f" Band: {band}"
        x, y = region_polygon.exterior.xy
        fig, ax = plt.subplots(figsize=(8, 6))
    
        if calexps!=None:
            if isinstance(calexps, int):
                n_cal = calexps
                calexps = self.datasetRefs[:n_cal]
            else:
                calexps = self.datasetRefs
            
            if band!= None:
                calexps = [dataRef for dataRef in calexps if dataRef.dataId["band"]==band]
            ok = True
            for dataRef in tqdm(calexps, desc="Plotting calexps"):
                data_id = {"visit":dataRef.dataId["visit"], "detector":dataRef.dataId["detector"]}
                calexp = Calexp(data_id)
                ra_corners, dec_corners = calexp.get_corners() 
                polygon = Polygon(zip(ra_corners, dec_corners))
                x_poly, y_poly = polygon.exterior.xy
                if ok:
                    ax.fill(x_poly, y_poly, color='gray', alpha=0.05, label="calexp", zorder=1)
                    ax.plot(x_poly, y_poly, color='gray', alpha=0.5, linewidth=lwC, zorder=1) 
                    ok = False
                else:
                    ax.fill(x_poly, y_poly, color='gray', alpha=0.05, zorder=1)  
                    ax.plot(x_poly, y_poly, color='gray', alpha=0.5, linewidth=lwC, zorder=1)  
        
        ax.plot(x, y, color="r", linewidth=lwT, label=label, linestyle="--", zorder=2)  

        if inj_points:
            cmap = cm.Blues  
            norm = Normalize(vmin=min(inj_points), vmax=max(inj_points)) 
            colors = cmap(norm(inj_points)) 
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
            cbar.set_label('Injected Points Intensity')
        else:
            colors = "blue"
        scatter = ax.scatter(ra_vals, dec_vals, alpha=0.6, color=colors, label=f"Injected points", edgecolor='k', zorder=3)
        
        ax.set_xlabel("ra (deg)")
        ax.set_ylabel("dec (deg)")
        ax.set_title(title)
        plt.legend(loc=(1.06,1.06))
        plt.grid(True)
        if band==None:
            plt.savefig(self.main_path+"/sky_map.png")
        else:
            plt.savefig(self.main_path+f"/sky_map_{band}.png")
        plt.show()
        self.log_task("Plotting sky map", det=band)

    def time_analysis(self, path=None):
        if path != None:
            df = pd.read_csv(path)
        else:
            df = self.log
        
        times = df["time"]
        task_names = df["task"][1:]
        details = df["detail"][1:]
        duration = [j - i for i, j in zip(times[:-1], times[1:])][1:]
        unique_tasks = sorted(set(task_names))
        cmap = plt.get_cmap("Pastel1")
        col_task = {task: cmap(i / len(unique_tasks)) for i, task in enumerate(unique_tasks)}
        task_colors = [col_task[task] for task in task_names[:-1]]
        plt.figure(figsize=(10, 6))
        plt.xlim(0, len(duration))
        plt.ylim(0, max(duration) * 1.1)

        x_positions = np.arange(len(duration))
        bar_width = 1
        bars = plt.bar(x_positions, duration, color=task_colors, width=bar_width)
        
        for i, (bar, detail, task) in enumerate(zip(bars, details, task_names)):
            try:
                plt.text(bar.get_x() + bar.get_width() * 0.5, bar.get_height() * 1.03, 
                         f'{int(detail)}', ha='center', va='center', fontsize=10, color='black')
            except:
                pass
        
        for task in unique_tasks:
            plt.bar(0, 0, color=col_task[task], label=task)
        
        plt.xlabel("Step")
        plt.ylabel("Duration (seconds)")
        plt.legend(title=f'Tasks - Duration: {np.sum(duration) / 60:.2f} min', 
                   ncol=min([6, len(unique_tasks) // 2]), loc=(0, 1.01))

        plt.savefig(f'{self.main_path}/time_analysis.png', bbox_inches='tight')
        plt.show()

    def plot_event(self, events_id = None, path = None):
        if events_id == None:
           events_id = set(self.data_events["event_id"].values)
        for i in events_id:
            lc_list = [lc for lc in self.inj_lc if int(lc.event_id)==i]    
            for j, lc in enumerate(lc_list):
                if j==5:
                    show=True
                lc.plot(title = lc_path, show=show)
            show=False

    def save_lc(self):
        for band in self.bands:
            lcs = [lc for lc in self.inj_lc if lc.band == band]
            print(f"Saving {len(lcs)} light curves in band {band}")
            for lc in tqdm(lcs):
                lc.data=lc.data[~pd.isna(lc.data["mag"])]
                lc.save(self.main_path+f"/lc_{lc.event_id}_{lc.band}.csv")

    def save_time_log(self):
        pd.DataFrame(self.log).to_csv(self.main_path+'/time_log.csv', index=False)


def plot_event(path, events_id=None, model="Pacz", mag_lim=(30,14), figsize=(10,4), join=True):
    mag_lim = mag_lim if ~join else None
    if events_id is None:
        events_id = set(pd.read_csv(path+"data_events.csv")["event_id"].values)
    show = False if join else True
    data_event = pd.read_csv(path+"data_events.csv")
    for i in events_id:
        lc_list = sorted([file for file in os.listdir(path) if file.startswith("lc") and file.split("_")[1] == str(i)])
        for j, lc_path in enumerate(lc_list):
            lc = LightCurve(data=pd.read_csv(path+lc_path))
            lc.band = lc_path.split(".")[-2][-1]
            lc.model = model
            data_lc = data_event[(data_event["event_id"]==i) & (data_event["band"]==lc.band)]
            t_0, t_E, u_0, m_base = data_lc[["t_0", "t_E", "u_0", "m_base"]].values[0]
            lc.params = {}
            for key, val in zip(["t_0", "t_E", "u_0", "m_base"], [t_0, t_E, u_0, m_base]):
                lc.params[key] = val
            if j==5 and join:
                show=True
            lc.plot(title = f"Event ID: {i}", mag_lim = mag_lim, figsize=figsize, show=show)
        show=False if join else True
    # def characterize_task(self, psf_iter = 1): 
    #     self.name = "Characterize"
    #     config = CharacterizeImageTask.ConfigClass()
    #     config.psfIterations = psf_iter
    #     self.task = CharacterizeImageTask(config=config)

    # def deblend_task(self):
    #     self.name = "Deblend"
    #     self.task = SourceDeblendTask(schema=self.schema)


