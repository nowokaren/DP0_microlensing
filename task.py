dp0_limits = [[48, 76], [-44,-28]] # [ra_lim, dec_lim]
import random
import time
from tqdm.notebook import tqdm
import os
import gc
from datetime import datetime
import numpy as np
from matplotlib.image import imread
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
    def __init__(self, name=None, ra=None, dec=None, bands=None, calexps_method="overlap", 
                 measure_method="ForcedMeas", density=None, area=None, radius=None, n_lc=0, 
                 main_path="./runs", data_events=None, data_calexps=None, scale=None):

        self.name = name if name else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_path = os.path.join(main_path, self.name)
        os.makedirs(self.main_path, exist_ok=True)  
        from_path = all(arg is None for arg in [ra, dec, bands, calexps_method, measure_method, density, area, radius, n_lc, data_events, data_calexps, scale])

        if from_path:
            log_path = os.path.join(self.main_path, f"{self.name}_log.txt")
            os.path.exists(log_path)
            print(f"Loading data from {log_path}")
            self._load_from_path(log_path)
            data_events = os.path.join(self.main_path, "data_events.csv")
            data_calexps = os.path.join(self.main_path, "data_calexps.csv")
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
        self.data_calexps = self._load_dataframe(data_calexps, columns=["detector", "visit", "mjd", "band", "overlap", "lc_ids"])
        
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
        self.bound_circle_radius = pixelization.triangle(self.htm_id).getBoundingCircle().getOpeningAngle().asDegrees() 
        htm_triangle = pixelization.triangle(self.htm_id)

        tri_ra_dec = [(LonLat.longitudeOf(v).asDegrees(), LonLat.latitudeOf(v).asDegrees()) for v in htm_triangle.getVertices()]
        self.htm_vertex = tri_ra_dec
        self.area = np.pi / (2 * 4**(self.htm_level - 1)) * (180 / np.pi)**2  # Area in deg²

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
                f"Scale: {self.scale:.3f} \n"
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
                f"Scale: {self.scale:.3f} \n"
                f"Number of LightCurves to inject: {self.n_lc}\n"
                f"Main Path: {self.main_path}\n"
                f"{separator}")


    def mjd(self, dataId):
        return self.data_calexps.loc[(self.data_calexps["visit"] == dataId["visit"]) & (self.data_calexps["detector"] == dataId["detector"]), "mjd"].values[0]

    def add_data_calexp(self, data_id, column, value):
        self.data_calexps.loc[(self.data_calexps["detector"] == data_id["detector"]) & (self.data_calexps["visit"] == data_id["visit"]),column] = value

    def add_data_event(self, event_id, band, column, value):
        self.data_events.loc[(self.data_events["event_id"] == event_id) & (self.data_events["band"] == band),column] = value

        
    def log_task(self, name, det=None):
        self.log["time"].append(time.time())
        self.log["task"].append(name)
        self.log["detail"].append(det)

    def create_schema(self):
        schema = afwTable.SourceTable.makeMinimalSchema()
        schema.addField("coord_raErr", type="F", doc="Error in RA coordinate")
        schema.addField("coord_decErr", type="F", doc="Error in Dec coordinate")
        return schema

    def load_calexp(self, n_max=None, load_mjd=True):
        bands_str = f"({', '.join(map(repr, self.bands))})"
        print("Collecting calexps...")
        if self.calexps_method == "htm":
            print(f'(ra,dec) = ({self.ra}, {self.dec}) \nHTM_ID = {self.htm_id} - HTM_level={self.htm_level} (bounded by a circle of radius ~{self.bound_circle_radius*3600:0.2f} arcsec.)')
            self.datasetRefs = list(butler.registry.queryDatasets("calexp", htm20=self.htm_id, where=f"band IN {bands_str}"))
        elif self.calexps_method == "overlap":
            target_point = SpherePoint(Angle(self.ra, degrees), Angle(self.dec, degrees))
            RA = target_point.getLongitude().asDegrees()
            DEC = target_point.getLatitude().asDegrees()
            circle = Region.from_ivoa_pos(f"CIRCLE {RA} {DEC} {self.radius}")
            self.datasetRefs = butler.query_datasets("calexp", where=f"visit_detector_region.region OVERLAPS my_region AND band IN {bands_str}", 
                                                     bind={"ra": RA, "dec": DEC, "my_region": circle}, limit=100000000)
            print(f'(ra,dec) = ({self.ra}, {self.dec}) \nCircle of radius ~{self.radius:0.3f} deg.')

        if load_mjd:
            if n_max is not None:
                datasetRefs = self.datasetRefs[:n_max]
            n_dataref = len(datasetRefs)
            print(f"Found {n_dataref} calexps.")
            ccd_visit = butler.get('ccdVisitTable')
            self.data_calexps["detector"] = [calexp_data.dataId['detector'] for calexp_data in datasetRefs]
            self.data_calexps["visit"] = [calexp_data.dataId['visit'] for calexp_data in datasetRefs]
            self.data_calexps["mjd"] = [
                ccd_visit[(ccd_visit['visitId'] == calexp_data.dataId['visit']) & 
                          (ccd_visit['detector'] == calexp_data.dataId['detector'])]['expMidptMJD'].values[0]
                for calexp_data in tqdm(datasetRefs, desc="Saving MJD values")
            ]
            self.data_calexps["band"] = [calexp_data.dataId['band'] for calexp_data in datasetRefs]
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

    def create_events_catalog(self, lc_data = "random"):
        if lc_data == "random":
            for event_id, m in enumerate(tqdm(np.linspace(17,22,process.n_lc), desc="Loading events")):
                lc_ra, lc_dec = process.generate_location()
                params = {"t_0": random.uniform(60300,61500),
               "t_E": random.uniform(20, 200), 
               "u_0": random.uniform(0.1,1)}
                for band, dm in zip(process.bands, delta_mag):
                    params["m_base"]=m+dm
                    process.add_lc(lc_ra, lc_dec, params, event_id=event_id, band=band)
        elif lc_data == "data_events":
            for i, (j, event) in enumerate(tqdm(process.data_events.iterrows(), desc="Loading events:")):
                lc_ra, lc_dec = event["ra"], event["dec"]
                params = {"t_0": event["t_0"],
                       "t_E": event["t_E"], 
                       "u_0": event["u_0"]}
                params["m_base"]=event["m_base"]
                process.add_lc(lc_ra, lc_dec, params, event_id=event["event_id"], band=event["band"])
        elif lc_data.endswith(".csv"):
            df = pd.read_csv(lc_data)
            for i, (j, event) in enumerate(tqdm(process.data_events.iterrows(), desc="Loading events:")):
                lc_ra, lc_dec = event["ra"], event["dec"]
                params = {"t_0": event["t_0"],
                       "t_E": event["t_E"], 
                       "u_0": event["u_0"]}
                params["m_base"]=event["m_base"]
                process.add_lc(lc_ra, lc_dec, params, event_id=event["event_id"], band=event["band"])            

    def add_lc(self, ra, dec, params, event_id, band, model="Pacz", plot=False, to_df = True):
        lc = LightCurve(ra, dec, band=band)
        lc.data["mjd"] = self.data_calexps[self.data_calexps["band"]==band]["mjd"]
        lc.data["visit"] = self.data_calexps[self.data_calexps["band"]==band]["visit"]
        lc.data["detector"] = self.data_calexps[self.data_calexps["band"]==band]["detector"]
        lc.simulate(params, model=model, plot=plot)
        lc.event_id = event_id
        new_lc = {"event_id":event_id, "ra": ra, "dec": dec, "model": model, "band": band}
        if to_df:
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
        print("Creating injection_catalog ...")
        catalog = []
        visit = calexp.data_id["visit"]
        detector = calexp.data_id["detector"]
        mjd = self.data_calexps[(self.data_calexps["visit"] == visit) & (self.data_calexps["detector"] == detector)]["mjd"].values[0]
        for i, lc in enumerate(self.inj_lc):
            if (lc.band == band): #and (calexp.contains(lc.ra, lc.dec)) and (~calexp.check_edge(lc.ra, lc.dec, d=100)):
                mag_inj = lc.data[(lc.data["visit"] == visit) & (lc.data["detector"] == detector)]["mag_inj"].values[0]
                catalog.append([i, visit, detector, lc.ra, lc.dec, "Star", mjd, mag_inj])
        if len(catalog)==0:
            return False
        else:
            return Table(rows=catalog,names=["lc_id", "visit", "detector", "ra", "dec", "source_type", "exp_midpoint", "mag"])

    # def check_injection_catalog(self, calexp, catalog, before_injection = True):
    #     ra, dec = catalog["ra"], catalog["dec"]
    #     if before_injection:
    #         mask_visit = np.array(catalog["visit"] == calexp.data_id["visit"])
    #         mask_detector = np.array(catalog["detector"] == calexp.data_id["detector"])
    #         mask_contain = np.array(calexp.contains(ra, dec)) 
    #         if False in mask_contain:
    #             print("Light curves NOT contained: ", len([i for i in range(len(ra)) if not mask_contain[i]]))
    #         mask_edge = np.array([calexp.check_edge(r, d, d=100) for r, d in zip(ra, dec)])
    #         if True in mask_edge:
    #             print("Light curves near edge: ", len([i for i in range(len(ra)) if mask_edge[i]]))
    #         keep_mask = mask_contain & ~mask_edge & mask_detector & mask_visit
    #     else:
    #         mask_flag = np.array([i!=0 for i in catalog["injection_flag"]])
    #         if True in mask_flag:
    #             print("Light curves marked FLAG: ", [i for i in range(len(ra)) if mask_flag[i]])
    #         keep_mask = ~mask_flag
    #     filtered_catalog = catalog[keep_mask]
    #     data_id = calexp.data_id
    #     self.data_calexps.loc[(self.data_calexps["detector"] == data_id["detector"]) & (self.data_calexps["visit"] == data_id["visit"]), "ids_events"] = "-".join(map(str, self.data_events[self.data_events["ra"].isin(filtered_catalog["ra"].value)].index))
    #     return filtered_catalog


        
    def inject_calexp(self, calexp, inject_table, save_fit = None):
        print("Injecting calexp...")
        exposure = calexp.expF
        try:
            injected_output = self.tasks["Injection"].run(
                injection_catalogs=[inject_table],
                input_exposure=exposure.clone(),
                psf=exposure.getPsf(),
                photo_calib=exposure.getPhotoCalib(),
                wcs=calexp.wcs)
        except Exception as e:
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


    def measure_calexp(self, schema, calexp, injected_catalog):
        table = pd.DataFrame(columns=["lc_id", "ra", "dec", "flux", "flux_err", "mag", "mag_err", "mag_inj", "flag"])
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
            sources["coord_ra"] = injected_catalog["ra"]
            sources["coord_dec"] = injected_catalog["dec"]
            sources["lc_id"] = injected_catalog["lc_id"]
            sources["mag_inj"] = injected_catalog["mag"]
            table["flux"], table["flux_err"] = sources["base_PsfFlux_instFlux"], sources["base_PsfFlux_instFluxErr"]
            MAGS = [calexp.get_mag(f, ferr) for f, ferr in zip(table["flux"], table["flux_err"])]
            table["mag"], table["mag_err"] = [M[0] for M in MAGS], [M[1] for M in MAGS] 
            table["ra"] = injected_catalog["ra"]; table["dec"] = injected_catalog["dec"]
            table["lc_id"] = sources["lc_id"]
            table["mag_inj"] = sources["mag_inj"]
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
                # table.loc[(table["ra"] == ra) & (table["dec"] == dec), ["flux", "flux_err", "mag", "mag_err", "flag"]] = [flux, flux_err, mag, mag_err, flag]
                table.loc[np.isclose(table["ra"], ra,rtol=1e-5) & np.isclose(table["dec"], dec,rtol=1e-5), ["flux", "flux_err", "mag", "mag_err", "flag"]] = [flux, flux_err, mag, mag_err, flag]

        flag_cols = [col for col in sources.columns if "flag" in col]
        mask = np.array([True in i for i in sources[flag_cols]])
        sources[mask][flag_cols]
        table["flag"] = ["-".join([col for col, val in zip(flag_cols, s) if val]) 
            for s in sources[flag_cols].as_array()]
        table.loc[table["flag"]=="","flag"] = 0
        return table, sources


    def sky_map(self, color='red', band=None, lwT=1, lwC=1, calexps=None, inj_points=True, show=True, ax=None):
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
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
    
        if calexps is not None:
            if isinstance(calexps, int) and not isinstance(calexps, bool):
                n_cal = calexps
                calexps = self.datasetRefs[:n_cal]
            else:
                calexps = self.datasetRefs
            desc="Plotting calexps"
            if band!= None:
                calexps = [dataRef for dataRef in calexps if dataRef.dataId["band"]==band]
                desc+=f" for band: {band}"
            ok = True
            for dataRef in tqdm(calexps, desc=desc):
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
        if calexps is not None:
            title += f" n_calexps: {len(calexps)}"
        ax.set_title(title)
        plt.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.21))
        plt.grid(True)
        if band==None:
            plt.savefig(self.main_path+"/sky_map.png")
        else:
            plt.savefig(self.main_path+f"/sky_map_{band}.png")
        

        self.log_task("Plotting sky map", det=band)
        if calexps is not None:
            result = len(calexps)
        else:
            result = None
        if show:
            plt.show()
        else:
            result = (result, ax)
        return result
        

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

    def save_light_curves(self, drop_na=True):
        for band in self.bands:
            lcs = [lc for lc in self.inj_lc if lc.band == band]
            print(f"Saving {len(lcs)} light curves for band {band}")
            for lc in tqdm(lcs):
                if drop_na:
                    lc.data=lc.data[~pd.isna(lc.data["mag"])]
                lc.save(self.main_path+f"/lc_{lc.event_id}_{lc.band}.csv")
                
    def plot_FoV(self, lc, calexp, r=40, fov_size=200, show=True):
        image_path = f"lc_{lc.event_id}_{lc.band}.png"
        if image_path not in os.listdir(self.main_path):
            roi = [(lc.ra, lc.dec), fov_size]
            ax = calexp.plot(roi=roi, figsize=(6,6))
            calexp.add_point(ax, lc.ra, lc.dec, r=r)
            calexp.save_plot(ax, self.main_path+"/"+image_path, show=show)
            print(f"{image_path} saved.")
            plt.clf() 
            plt.close(ax.figure) 
            plt.close('all') 
            gc.collect()
            del ax, calexp

    def save_data(self, drop_na=True):
        for lc in self.inj_lc:
            self.add_data_event(lc.event_id, lc.band, "points", lc.data["mag"].count())
        self.data_events.to_csv(self.main_path+'/data_events.csv', index=False)
        self.data_calexps.to_csv(self.main_path+'/data_calexps.csv', index=False)
        self.save_light_curves(drop_na=drop_na)
        pd.DataFrame(self.log).to_csv(self.main_path+'/time_log.csv', index=False)
        self.time_analysis()


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


def plot_lc_examples(run_path, events_id, name, join=False, plot_fov=True):
    show = False if join else True
    data_event = pd.read_csv(run_path + "data_events.csv")
    n_ev = len(events_id)
    num_rows = n_ev * 2 if plot_fov else n_ev  # Define el número de filas dinámicamente
    fig, axs = plt.subplots(num_rows, 6, figsize=(20, num_rows * 3))
    
    for i, event_id in enumerate(events_id):
        row_index = i * (2 if plot_fov else 1)  # Calcula la fila correctamente
        
        lc_list = sorted([file for file in os.listdir(run_path) if file.startswith("lc") and file.split("_")[1] == str(event_id) and file.endswith(".csv")])
        if len(lc_list)==0:
            print(f"There isn't event with id {event_id}. Skipping...")
            continue

        lc_list_ugrizy = []
        for band in "ugrizy":
            lc_band_path = [lc for lc in lc_list if f"_{band}" in lc][0]
            lc_list_ugrizy.append(lc_band_path)
        
        for j, (lc_path, ax) in enumerate(zip(lc_list_ugrizy, axs[row_index])):  
            lc = LightCurve(data=run_path + lc_path)
            lc.model = "Pacz"
            lc.event_id = event_id
            
            # Extraer parámetros del evento
            data_lc = data_event[(data_event["event_id"] == event_id) & (data_event["band"] == lc.band)]
            t_0, t_E, u_0, m_base = data_lc[["t_0", "t_E", "u_0", "m_base"]].values[0]
            lc.params = {key: val for key, val in zip(["t_0", "t_E", "u_0", "m_base"], [t_0, t_E, u_0, m_base])}
            
            plt.sca(ax)
            lc.plot(title=None, mag_lim=None, figsize=None, show=False) 
            ax.invert_yaxis()
            ax.grid()
            ax.legend().remove()
            ax.set_title('')
            ax.tick_params(axis='y', labelsize=8)
            ax.tick_params(axis='x', labelsize=8)
            ax.set_xlabel("Epoch (MJD)", fontsize=10)
            ax.text(0.02, 0.98, f"eventId: {event_id}\nBand: {lc.band}", transform=ax.transAxes, 
                    fontsize=10, verticalalignment='top', horizontalalignment='left')
            
            if i != n_ev - 1:
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_xlabel("")
            
            if j != 0:
                ax.set_ylabel("")
        
        if plot_fov:
            for j, (lc_path, img_ax) in enumerate(zip(lc_list_ugrizy, axs[row_index + 1])):  
                img_path = os.path.join(run_path, lc_path.split(".")[0] + ".png")
                if os.path.exists(img_path):
                    img = imread(img_path)
                    y_start, y_end = 30, -20
                    x_start, x_end = 40, -30
                    img_cropped = img[y_start:y_end, x_start:x_end]
                    img_ax.imshow(img_cropped)
                    band = lc_path.split(".")[0].split("_")[-1]
                    img_ax.set_title(f"Event {event_id} - Band: {band}", fontsize=10)
                    img_ax.set_xlabel("RA (deg)", fontsize=10)
                    img_ax.set_ylabel("Dec (deg)", fontsize=10)

                    # img_ax.tick_params(axis='both', which='both', length=0) 
                    # img_ax.spines['top'].set_visible(False)  
                    # img_ax.spines['right'].set_visible(False) 
                    # img_ax.spines['left'].set_visible(False) 
                    # img_ax.spines['bottom'].set_visible(False)

                    img_ax.set_xticks([])
                    img_ax.set_yticks([])
                    img_ax.set_xticklabels([])
                    img_ax.set_yticklabels([])

                    img_ax.spines['top'].set_visible(False)
                    img_ax.spines['right'].set_visible(False)
                    img_ax.spines['left'].set_visible(False)
                    img_ax.spines['bottom'].set_visible(False)

                else:
                    img_ax.axis("off")  

    plt.subplots_adjust(hspace=0.00001, wspace=0.05) 
    plt.tight_layout(pad=0.000001)
    plt.savefig(run_path + name, bbox_inches='tight')



