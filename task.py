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

class Run:
    def __init__(self, name=None, main_path = "runs/", htm_level=20):#, schema=None):
        date = datetime.now()
        self.name = name if name else date.strftime("%Y%m%d_%H%M%S")
        self.main_path = main_path + self.name + "/"
        os.makedirs(self.main_path, exist_ok=True)
        self.tasks = {}
        self.inj_lc = []
        self.log = {"task":["Start"], "time":[time.time()], "detail":[None]}
        self.calexp_data_ref = None
        self.mjds = None
        self.htm_level = htm_level
        self.inject_table = None
        # self.schema = self.create_schema()
        # self.tab = afwTable.SourceTable.make(self.schema)
        # self.visits = None
        # self.detectors = None

    def log_task(self, name, det=None):
        self.log["time"].append(time.time())
        self.log["task"].append(name)
        self.log["detail"].append(det)

    def create_schema(self):
        schema = afwTable.SourceTable.makeMinimalSchema()
        schema.addField("coord_raErr", type="F", doc="Error in RA coordinate")
        schema.addField("coord_decErr", type="F", doc="Error in Dec coordinate")
        return schema
        
    def add_lc(self, params, model="Pacz",  ra=None, dec=None, dist=0.5, plot=False):
        if not ra and not dec:
            if len(self.inj_lc) == 0:
                ra = random.uniform(dp0_limits[0][0], dp0_limits[0][1])
                dec = random.uniform(dp0_limits[1][0], dp0_limits[1][1])
            else:
                first_lc = self.inj_lc[0]
                f_ra = first_lc.ra; f_dec = first_lc.dec
                ra = random.uniform(f_ra-dist, f_ra+dist)
                dec = random.uniform(f_dec-dist, f_dec+dist)
        lc = LightCurve(ra, dec)
        if len(self.inj_lc) == 0:
            lc.collect_calexp(self.htm_level)
            self.calexp_data_ref = lc.calexp_data_ref
            self.mjds = lc.data["mjd"]
            self.calexp_dataIds =  [{"visit": dataref.dataId["visit"], "detector":dataref.dataId["detector"]} for dataref in self.calexp_data_ref]
        else:
            lc.data["mjd"] = self.mjds
            lc.data["visit"] = self.inj_lc[0].data["visit"]
            lc.data["detector"] = self.inj_lc[0].data["detector"]
        lc.simulate(params, model=model, plot=plot)
        self.inj_lc.append(lc)

    def inject_task(self):
        inject_config = VisitInjectConfig()
        self.tasks["Injection"] = VisitInjectTask(config=inject_config)
        
    def inject_calexp(self, calexp, save_fit = None):
        '''Creates injecting catalog and inject light curve's points if the calexp contains it.
        save_fit = name of the file to be saved'''
        inj_lightcurves = []
        for i, lc in enumerate(self.inj_lc):
            if calexp.contains(lc.ra, lc.dec):
                aux = []
                data = [i, calexp.calexp_data["visit"], calexp.calexp_data["detector"], lc.ra, lc.dec,"Star", self.mjds[i], lc.data["mag_sim"][i]]
                for item in data:
                    aux.append([item])
                if len(inj_lightcurves) == 0:
                    inject_table =  Table(aux, names=['injection_id', 'visit', 'detector', 'ra', 'dec', 'source_type', 'exp_midpoint', 'mag'])
                else:
                    inject_table = vstack([inject_table, Table(aux, names=['injection_id', 'visit', 'detector', 'ra', 'dec', 'source_type', 'exp_midpoint', 'mag'])])
                inj_lightcurves.append(i)
        if len(inj_lightcurves)>0:
            print(f"Points injected: {len(inj_lightcurves)}")
            print(inj_lightcurves)
            exposure = calexp.expF
            injected_output = self.tasks["Injection"].run(
                injection_catalogs=[inject_table],
                input_exposure=exposure.clone(),
                psf=exposure.getPsf(),
                photo_calib=exposure.getPhotoCalib(),
                wcs=calexp.wcs)
            injected_exposure = injected_output.output_exposure
            injected_catalog = injected_output.output_catalog
            self.log_task("Injection")

            if save_fit is not None:
                injected_exposure.writeFits(self.main_path+save_fit)
            if self.inject_table == None: 
                self.inject_table = injected_catalog
            else:
                self.inject_table = vstack([self.inject_table, injected_catalog])
            return injected_exposure, inj_lightcurves
            self.log_task("Saving injection results")
        else:
            print("No point is contained in the calexp")
            return None, None


    def measure_task(self):
        schema = afwTable.SourceTable.makeMinimalSchema()
        raerr = schema.addField("coord_raErr", type="F")
        decerr = schema.addField("coord_decErr", type="F")
        algMetadata = dafBase.PropertyList()
        config = SourceDetectionTask.ConfigClass()
        config.thresholdValue = 4
        config.thresholdType = "stdev"
        self.tasks["Detection"] = SourceDetectionTask(schema=schema, config=config)
        del config
        config = SingleFrameMeasurementTask.ConfigClass()
        self.tasks["Measurement"] = SingleFrameMeasurementTask(schema=schema,
                                                           config=config,
                                                           algMetadata=algMetadata)
        del config
        return schema

    def measure_calexp(self, calexp, schema):
        tab = afwTable.SourceTable.make(schema)
        del schema
        result = self.tasks["Detection"].run(tab, calexp)
        sources = result.sources
        self.log_task("Detection", det=len(sources))
        self.tasks["Measurement"].run(measCat=sources, exposure=calexp)
        self.log_task("Measurement")
        return sources


    def find_flux(self, sources, inj_lc, save=None, search_factor=1):
        fluxes = []; fluxes_err = []
        # for i, lc in enumerate(self.inj_lc):
            # count = 0; point = []
            # while len(point)!=1:
            #     point = sources[abs(sources["coord_ra"]-lc.ra*np.pi/180)< 0.0000001*search_factor]
            #     if len(point)>1:
            #         search_factor*=3/4
            #     elif len(point)==0:
            #         search_factor*=4/3
            #     count+=1

        sources_coords = np.column_stack((sources["coord_ra"], sources["coord_dec"]))
        tree = KDTree(sources_coords)
        initial_radius = 1e-7  # Radio inicial de búsqueda en radianes (~0.02 arcsec)
        min_radius = 1e-10 
        max_radius = 1e-5 
        for i, lc in enumerate(self.inj_lc):
            if i in inj_lc:
                target_coords = [lc.ra*np.pi/180, lc.dec*np.pi/180]
                search_radius = initial_radius
                count = 0    
                while True:
                    idx = tree.query_ball_point(target_coords, r=search_radius)        
                    if len(idx) == 1:
                        break
                    elif len(idx) > 1:
                        search_radius *= 0.95
                    else:
                        search_radius *= 1.05
                    search_radius = max(min_radius, min(search_radius, max_radius))
                    count += 1
                    if count > 1000:
                        print(f"No se pudo encontrar un único punto después de 1000 iteraciones para la curva de luz {i}.")
                        continue
                point = sources[idx[0]]
                flux = point["base_PsfFlux_instFlux"]; flux_err = point["base_PsfFlux_instFluxErr"]
                print(f"Founded point in source table in {count} iteration/s")
                fluxes.append(flux); fluxes_err.append(flux_err)
                if save != None:
                    lc.add_flux(flux, flux_err, save)
                else:
                    fluxes.append(np.nan); fluxes_err.append(np.nan)
        self.log_task("Finding points", det = len(inj_lc))            
        return fluxes, fluxes_err  

    def time_analysis(self):
        times = self.log["time"]
        task_names = self.log["task"]
        duration = [j - i for i, j in zip(times[:-1], times[1:])]
        unique_tasks = sorted(set(task_names))
        cmap = plt.get_cmap("tab20")  
        col_task = {task: cmap(i / len(unique_tasks)) for i, task in enumerate(unique_tasks)}
        task_colors = [col_task[task] for task in task_names[:-1]]
        
        # plt.figure(figsize=(27, 6))
        # plt.xlim(0, 720)
        # plt.ylim(0, 15)
        plt.bar(range(len(duration)), duration, color=task_colors, width=2)
        for task in unique_tasks:
            plt.bar(0, 0, color=col_task[task], label=task)
        plt.xlabel("Step")
        plt.ylabel("Duration (seconds)")
        plt.legend(title=f'Tasks - Duration: {np.sum(duration) / 60:.2f} min', ncol=min([6, len(unique_tasks)/2]), loc=(0, 1.01))
        plt.savefig(f'{self.main_path}time_analysis.png', bbox_inches='tight')
                
    def save_lc(self):
        for i, lc in enumerate(self.inj_lc):
            lc.save(self.main_path+f"lc_{i}.cvs")

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

