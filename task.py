dp0_limits = [[48, 76], [-44,-28]] # [ra_lim, dec_lim]
import random
import time
from tqdm import tqdm
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

from tools import tri_sample, triangle_min_height, circ_sample

from lsst.daf.butler import Butler
# butler_config = 'dp02'
butler_config = 'dp02-direct'
collections = '2.2i/runs/DP0.2'
butler = Butler(butler_config, collections=collections)

class Run:
    def __init__(self, ra, dec, scale, band, name=None, main_path = "runs/", data_events = None, data_calexp = None, method = "htm"):
        self.ra = ra; self.dec = dec
        self.scale = scale # HTM level or radius in deg
        self.method = method
        self.band = band
        self.inj_lc = []
        date = datetime.now()
        self.name = name if name else date.strftime("%Y%m%d_%H%M%S")
        self.main_path = main_path + self.name + "/"
        os.makedirs(self.main_path, exist_ok=True)
        self.tasks = {}
        self.events = []
        self.log = {"task":["Start"], "time":[time.time()], "detail":[None]}
        if data_events is None:
            self.data_events = pd.DataFrame(columns=["ra", "dec", "model", "params", "band", "points"]) 
        elif type(data_events)==str:
            self.data_events = pd.read_csv(data_events)
        else:
            self.data_events = data_events
            
        if data_calexp is None:
            self.data_calexp = pd.DataFrame(columns=["detector", "visit", "mjd", "band"]) 
        elif type(data_calexp)==str:
            self.data_calexp = pd.read_csv(data_calexp)
        else:
            self.data_calexp = data_calexp
        self.datasetRefs = None
        self.dist = None
        self.ref_dist = None
        self.inject_table = None
        if self.method == "htm":
            self.htm_level = scale
            self.htm_id = None
            self.htm_vertex = None
        elif self.method == "overlap":
            self.radius = scale

    @property
    def center(self):
        lc = self.inj_lc[0]
        return lc.ra, lc.dec

    def log_task(self, name, det=None):
        self.log["time"].append(time.time())
        self.log["task"].append(name)
        self.log["detail"].append(det)

    def create_schema(self):
        schema = afwTable.SourceTable.makeMinimalSchema()
        schema.addField("coord_raErr", type="F", doc="Error in RA coordinate")
        schema.addField("coord_decErr", type="F", doc="Error in Dec coordinate")
        return schema

    def collect_calexp(self, method="htm"):
        if method == "htm":
            pixelization = HtmPixelization(self.scale)
            self.htm_id = pixelization.index(UnitVector3d(LonLat.fromDegrees(self.ra, self.dec)))
            circle = pixelization.triangle(self.htm_id).getBoundingCircle()
            bound_circle_radius = circle.getOpeningAngle().asDegrees()*3600
            level = pixelization.getLevel()
            htm_triangle = pixelization.triangle(self.htm_id)
            tri_ra_dec = []
            for vertex in htm_triangle.getVertices():
                lon = LonLat.longitudeOf(vertex).asDegrees()
                lat = LonLat.latitudeOf(vertex).asDegrees()
                tri_ra_dec.append((lon, lat))
            self.htm_vertex = tri_ra_dec
            print(f'(ra,dec) = ({self.ra}, {self.dec}) \nHTM_ID = {self.htm_id} - HTM_level={self.htm_level} (bounded by a circle of radius ~{bound_circle_radius:0.2f} arcsec.)')
            self.datasetRefs = list(butler.registry.queryDatasets("calexp", htm20=self.htm_id, where=f"band = '{self.band}'"))
        elif self.method == "overlap":
            target_point = SpherePoint(Angle(self.ra, degrees), Angle(self.dec, degrees))
            RA = target_point.getLongitude().asDegrees()
            DEC = target_point.getLatitude().asDegrees()
            circle = Region.from_ivoa_pos(f"CIRCLE {RA} {DEC} {self.scale}")
            self.datasetRefs = butler.query_datasets("calexp", where=f"visit_detector_region.region OVERLAPS my_region AND band = '{self.band}'", bind={"ra": RA, "dec": DEC, "my_region": circle})
            print(f'(ra,dec) = ({self.ra}, {self.dec}) \nCircle of radius ~{self.scale:0.2f} deg.)')
        print(f"Found {len(self.datasetRefs)} calexps.")
        ccd_visit = butler.get('ccdVisitTable')
        self.data_calexp["detector"] = [calexp_data.dataId['detector'] for calexp_data in self.datasetRefs]
        self.data_calexp["visit"] = [calexp_data.dataId['visit'] for calexp_data in self.datasetRefs]
        self.data_calexp["mjd"] = [ccd_visit[(ccd_visit['visitId'] == calexp_data.dataId['visit']) & (ccd_visit['detector'] == calexp_data.dataId['detector'])]['expMidptMJD'].values[0] for calexp_data in self.datasetRefs]

    def generate_location(self, dist=None):
        if dist is None:
            if self.method == "htm":
                self.ref_dist = triangle_min_height(self.htm_vertex)
            elif self.method == "overlap":
                self.ref_dist = self.radius
            self.dist = self.ref_dist/20
        else:
            self.dist = dist
        distance = self.dist/2
        while distance < self.dist:
            if self.method == "htm": 
                ra, dec = tri_sample(self.htm_vertex)
            elif self.method == "overlap":
                ra, dec = circ_sample(*self.center, self.radius)
            for lc in self.inj_lc:
                distance = np.sqrt((lc.ra-ra)**2+(lc.dec-dec)**2)
                if distance < self.dist:
                    break
        return ra, dec

    def add_lc(self, ra, dec, params, band, model="Pacz", plot=False):
        lc = LightCurve(ra, dec, band = band)
        lc.data["mjd"] = self.data_calexp["mjd"]
        lc.data["visit"] = self.data_calexp["visit"]
        lc.data["detector"] = self.data_calexp["detector"]
        lc.simulate(params, model=model, plot=plot)
        new_lc = {"ra": ra, "dec": dec, "model": model, "params": params, "band": band}
        self.data_events.loc[len(self.data_events)] = new_lc
        self.inj_lc.append(lc)

    def inject_task(self):
        inject_config = VisitInjectConfig()
        self.tasks["Injection"] = VisitInjectTask(config=inject_config)

    def create_injection_table(self):
        idx = np.arange(len(self.inj_lc))
        visits = self.data_calexp["visit"]
        detector = self.data_calexp["detector"]
        mjd = self.data_calexp["mjd"]
        ra = [lc.ra for lc in self.inj_lc]
        dec = [lc.dec for lc in self.inj_lc]
        star = ["Star" for lc in self.inj_lc]
        mag_sim = [lc.data["mag_sim"] for lc in self.inj_lc]
        data = []
        for _ in [idx, visits, detector, ra, dec, star, mjd, mag_sim]:
            data.append([_])
        return Table(data, names=['injection_id', 'visit', 'detector', 'ra', 'dec', 'source_type', 'exp_midpoint', 'mag'])
        
    def inject_calexp(self, inject_table, calexp): #data, save_fit = None):
        '''Creates injecting catalog and inject light curve's points if the calexp contains it.
        save_fit = name of the file to be saved'''
        # inj_lightcurves = []
        # for i, lc in enumerate(self.inj_lc):
        #     if calexp.contains(lc.ra, lc.dec):
        #         aux = []
        #         data = [i, calexp.data_id["visit"], calexp.data_id["detector"], lc.ra, lc.dec,"Star", self.mjds[i], lc.data["mag_sim"][i]]
        #         for item in data:
        #             aux.append([item])
        #         if len(inj_lightcurves) == 0:
        #             inject_table =  Table(aux, names=['injection_id', 'visit', 'detector', 'ra', 'dec', 'source_type', 'exp_midpoint', 'mag'])
        #         else:
        #             inject_table = vstack([inject_table, Table(aux, names=['injection_id', 'visit', 'detector', 'ra', 'dec', 'source_type', 'exp_midpoint', 'mag'])])
        #         inj_lightcurves.append(i)
        
        # n_inj = len(inj_lightcurves)
        # if len(inj_lightcurves)>0:
            # print(f"Points injected: {n_inj}")
            # print(inj_lightcurves)
        exposure = calexp.expF
        injected_output = self.tasks["Injection"].run(
            injection_catalogs=[inject_table],
            input_exposure=exposure.clone(),
            psf=exposure.getPsf(),
            photo_calib=exposure.getPhotoCalib(),
            wcs=calexp.wcs)
        injected_exposure = injected_output.output_exposure
        injected_catalog = injected_output.output_catalog
        self.log_task("Injection", det = n_inj)
        return injected_output
        #     if save_fit is not None:
        #         injected_exposure.writeFits(self.main_path+save_fit)
        #     if self.inject_table == None: 
        #         self.inject_table = injected_catalog
        #     else:
        #         self.inject_table = vstack([self.inject_table, injected_catalog])
        #     return injected_exposure, inj_lightcurves
        #     self.log_task("Saving injection results")
        # else:
        #     print("No point is contained in the calexp")
            # return None, None


    def measure_task(self):
        schema = afwTable.SourceTable.makeMinimalSchema()
        raerr = schema.addField("coord_raErr", type="F")
        decerr = schema.addField("coord_decErr", type="F")
        algMetadata = dafBase.PropertyList()
        config = SourceDetectionTask.ConfigClass()
        config.thresholdValue = 4
        config.thresholdType = "stdev"
        self.tasks["Detection"] = SourceDetectionTask(schema=schema, config=config)
        config = SingleFrameMeasurementTask.ConfigClass()
        self.tasks["Measurement"] = SingleFrameMeasurementTask(schema=schema,
                                                           config=config,
                                                           algMetadata=algMetadata)
        return schema

    def measure_calexp(self, calexp, schema):
        tab = afwTable.SourceTable.make(schema)
        result = self.tasks["Detection"].run(tab, calexp)
        sources = result.sources
        # sources = calexp.get_sources(self.tasks["Detection"], schema)
        self.log_task("Detection", det=len(sources))
        self.tasks["Measurement"].run(measCat=sources, exposure=calexp)
        self.log_task("Measurement")
        return sources


    def find_flux(self, sources, inj_lc, save=None, search_factor=1):
        fluxes = []; fluxes_err = []
        try:
            for i, lc in enumerate(tqdm(self.inj_lc, desc="Searching flux in source table")):
                if i in inj_lc:
                    ra_rad = Angle(lc.ra, degrees).asRadians(); dec_rad = Angle(lc.ra, degrees).asRadians()
                    near = np.argmin([SpherePoint(ra_rad,dec_rad, degrees).separation(SpherePoint(sources["coord_ra"][i],sources["coord_dec"][i], radians)) for i in range(len(sources))])
                    flux = sources["base_PsfFlux_instFlux"][near]; flux_err = sources["base_PsfFlux_instFluxErr"][near]
                    fluxes.append(flux); fluxes_err.append(flux_err)
                    if save != None:
                        lc.add_flux(flux, flux_err, save)
                    else:
                        fluxes.append(np.nan); fluxes_err.append(np.nan)
            self.log_task("Finding points", det = len(inj_lc)) 
        except KeyboardInterrupt:
            print(f'Searching in lc {i}')
        return fluxes, fluxes_err  

    def sky_map(self, color='red', lwT=1, lwC=1, calexps=True, inj_points=True):
        ra_vals = [lc.ra for lc in self.inj_lc]
        dec_vals = [lc.dec for lc in self.inj_lc]
        inj_points = [lc.data["mag"].count() for lc in self.inj_lc] 
        if self.method == "htm":
            label = f"HTM level {self.htm_level}"
            title = f"Injected sources distribution in the HTM triangle (Level {self.htm_level})"
            region_polygon = Polygon(self.htm_vertex)
        elif self.method == "overlap":
            label = f"circle of radius {self.radius}"
            title = f"Injected sources distribution in the circle of radius {self.radius}"
            ra,dec = self.inj_lc[0].ra, self.inj_lc[0].dec 
            center = Point(ra, dec)
            region_polygon = center.buffer(self.radius) 
        x, y = region_polygon.exterior.xy
        fig, ax = plt.subplots(figsize=(8, 6))
    
        if calexps:
            ok = True
            for dataRef in tqdm(self.calexp_data_ref, desc="Loading calexps"):
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

                # ax.scatter(ra_vals, dec_vals, color='blue', label=f"Injected points")  
        cmap = cm.Blues  
        norm = Normalize(vmin=min(inj_points), vmax=max(inj_points)) 
        colors = cmap(norm(inj_points)) 
        scatter = ax.scatter(ra_vals, dec_vals, color=colors, label=f"Injected points", edgecolor='k', zorder=3)
        if inj_points:
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
            cbar.set_label('Injected Points Intensity')
        
        ax.set_xlabel("ra (deg)")
        ax.set_ylabel("dec (deg)")
        ax.set_title(title)
        plt.legend(loc=(1.06,1.06))
        plt.grid(True)
        plt.savefig(self.main_path+"sky_map.png")
        plt.show()
        self.log_task("Plotting sky map")


    # def time_analysis(self):
    #     times = self.log["time"][1:]
    #     task_names = self.log["task"][1:]
    #     details = self.log["detail"][1:]
    #     duration = [j - i for i, j in zip(times[:-1], times[1:])]
    #     unique_tasks = sorted(set(task_names))
    #     cmap = plt.get_cmap("tab20")
    #     col_task = {task: cmap(i / len(unique_tasks)) for i, task in enumerate(unique_tasks)}
    #     task_colors = [col_task[task] for task in task_names[:-1]]
        
    #     plt.figure(figsize=(27, 6))
        
    #     # Define límites del gráfico
    #     plt.xlim(0, len(duration))  # Basado en la cantidad de pasos
    #     plt.ylim(0, max(duration) * 1.1)  # Agrega espacio para las etiquetas
        
    #     # Graficar las barras
    #     bars = plt.bar(range(len(duration)), duration, color=task_colors, width=2)
        
    #     # Agregar etiquetas de `detail` sobre cada barra
    #     for i, (bar, detail) in enumerate(zip(bars, details)):
    #         if detail is not None:  # Solo agrega texto si `detail` no es None
    #             plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.8, 
    #                      f'{detail}', ha='center', va='center', fontsize=8, color='black')
        
    #     # Agregar leyenda
    #     for task in unique_tasks:
    #         plt.bar(0, 0, color=col_task[task], label=task)
        
    #     # Configuración de etiquetas y leyenda
    #     plt.xlabel("Step")
    #     plt.ylabel("Duration (seconds)")
    #     plt.legend(title=f'Tasks - Duration: {np.sum(duration) / 60:.2f} min', 
    #                ncol=min([6, len(unique_tasks) // 2]), loc=(0, 1.01))
        
    #     # Guardar el gráfico
    #     plt.savefig(f'{self.main_path}time_analysis.png', bbox_inches='tight')
    #     plt.show()

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
        # cmap = plt.get_cmap("tab20")
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
            if str(detail) != "nan" and detail is not None and task!= "Finding points":  
                plt.text(bar.get_x() + bar.get_width() * 0.5, bar.get_height() * 1.03, 
                         f'{int(detail)}', ha='center', va='center', fontsize=10, color='black')
        
        for task in unique_tasks:
            plt.bar(0, 0, color=col_task[task], label=task)
        
        plt.xlabel("Step")
        plt.ylabel("Duration (seconds)")
        plt.legend(title=f'Tasks - Duration: {np.sum(duration) / 60:.2f} min', 
                   ncol=min([6, len(unique_tasks) // 2]), loc=(0, 1.01))

        plt.savefig(f'{self.main_path}time_analysis.png', bbox_inches='tight')
        plt.show()

    def save_lc(self):
        # for band in self.bands:
        #     lcs = [lc for lc in self.inj_lc if lc.band == band]
        for i, lc in enumerate(self.inj_lc):
            lc.save(self.main_path+f"lc_{i}_{lc.band}.csv")

    def save_time_log(self):
        pd.DataFrame(self.log).to_csv(self.main_path+'time_log.csv', index=False)


    # def detect_measure_calexp(self, exposure):
    #     detect_result = self.tasks["Detection"].run(self.tab, exposure)
    #     self.tasks["Measurement"].run(measCat=detect_result.sources, exposure=exposure) 
        


    # def detection_task(self, threshold = 5, threshold_type = "stdev", schema = None):
    #     config = SourceDetectionTask.ConfigClass()
    #     config.thresholdValue = threshold
    #     config.thresholdType = threshold_type
    #     self.tasks["Detection"]=SourceDetectionTask(schema=self.schema, config=config)
    #     return schema

    # def measure_task(self, plugins=None, name = "Measurement", schema = None):
    #     '''plugin example = "base_SkyCoord", "base_PsfFlux", 'base_SdssCentroid' (just excecute these plugin measurment)'''
    #     config = SingleFrameMeasurementTask.ConfigClass()
    #     algMetadata = dafBase.PropertyList()
    #     if schema is None:
    #         schema = self.schema
    #     if plugins:
    #         for plugin_name in config_coord.plugins.keys():
    #             config.plugins[plugin_name].doMeasure = False
    #         for plugin in plugins:
    #             config.plugins[plugin].doMeasure = True
    #     self.tasks[name]= SingleFrameMeasurementTask(schema=schema,config=config)
    #     return schema

    # def detect_measure_task(self, threshold = 5, threshold_type = "stdev", meas_plugins=None, meas_name = "Measurement"):
    #     schema = afwTable.SourceTable.makeMinimalSchema()
    #     algMetadata = dafBase.PropertyList()
    #     # schema.addField("coord_raErr", type="F", doc="Error in RA coordinate")
    #     # schema.addField("coord_decErr", type="F", doc="Error in Dec coordinate")
        
    #     config = SourceDetectionTask.ConfigClass()
    #     config.thresholdValue = threshold
    #     config.thresholdType = threshold_type
    #     self.tasks["Detection"]=SourceDetectionTask(schema=schema, config=config)
    #     config = SingleFrameMeasurementTask.ConfigClass()
    #     if meas_plugins:
    #         for plugin_name in config_coord.plugins.keys():
    #             config.plugins[plugin_name].doMeasure = False
    #         for plugin in plugins:
    #             config.plugins[plugin].doMeasure = True
    #     self.tasks[meas_name]= SingleFrameMeasurementTask(schema=schema, config=config, algMetadata=algMetadata)
    #     return SourceDetectionTask(schema=schema, config=config), SingleFrameMeasurementTask(schema=schema, config=config, algMetadata=algMetadata)



    # def characterize_task(self, psf_iter = 1): 
    #     self.name = "Characterize"
    #     config = CharacterizeImageTask.ConfigClass()
    #     config.psfIterations = psf_iter
    #     self.task = CharacterizeImageTask(config=config)

    # def deblend_task(self):
    #     self.name = "Deblend"
    #     self.task = SourceDeblendTask(schema=self.schema)

    # def forced_measure_task(self, ):
    #     schema = self.schema #afwTable.SourceTable.makeMinimalSchema()
    #     alias = schema.getAliasMap() 
    #     x_key = schema.addField("centroid_x", type="D")
    #     y_key = schema.addField("centroid_y", type="D")
    #     alias.set("slot_Centroid", "centroid")
        
    #     xx_key = schema.addField("shape_xx", type="D")
    #     yy_key = schema.addField("shape_yy", type="D")
    #     xy_key = schema.addField("shape_xy", type="D")
    #     alias.set("slot_Shape", "shape")
    #     type_key = schema.addField("type_flag", type="F")
    #     config = ForcedMeasurementTask.ConfigClass()
    #     config.copyColumns = {}
    #     config.plugins.names = ['base_SdssCentroid', "base_TransformedCentroid",
    #         "base_PsfFlux",
    #         "base_TransformedShape",
    #     ]
    #     config.doReplaceWithNoise = False
    #     self.tasks["ForcedMeasurement"] = ForcedMeasurementTask(schema, config=config)
    #     for j, contained in enumerate(ok):
    #         if contained:
    #             rai, deci = locs[j]
    #             sourceRec = forcedSource.addNew()
    #             coord = geom.SpherePoint(np.radians(rai) * geom.radians,
    #                                      np.radians(deci) * geom.radians)
    #             sourceRec.setCoord(coord)
    #             sourceRec[x_key] = pix[j][0]
    #             sourceRec[y_key] = pix[j][1]
    #         # sourceRec[type_key] = 1
    #     # print(forcedSource.asAstropy()[["coord_ra", "coord_dec", "centroid_x", "centroid_y"]])
    #     forcedMeasCat = forcedMeasurementTask.generateMeasCat(calexp, forcedSource, calexp.getWcs())
    #     forcedMeasurementTask.run(forcedMeasCat, calexp, forcedSource, calexp.getWcs())
    #     sources = forcedMeasCat.asAstropy()


    # def run(self, name, **kwargs):
    #     if name in self.tasks and hasattr(self.tasks[name], "run"):
    #         result = self.tasks[name].run(**kwargs)
    #         self.log_task(task_name)
    #         return result
    #     else:
    #         print(f"Task '{name}' isn't configured or it hasn't 'run' method.")
    #     return None


