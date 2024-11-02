dp0_limits = [[48, 76], [-44,-28]] # [ra_lim, dec_lim]
import random
import time
import os
from datetime import datetime
import numpy as np
import pandas as pd
from astropy.table import Table, vstack
from lsst.afw import table as afwTable
from lsst.meas.base import SingleFrameMeasurementTask
from lsst.meas.algorithms import SourceDetectionTask
from light_curves import LightCurve
from exposures import Calexp

class Run:
    def __init__(self, name=None, main_path = "runs/", htm_level=20):#, schema=None):
        date = datetime.now()
        self.name = date.strftime("%Y%m%d_%H%M%S")
        self.main_path = main_path + self.name + "/"
        os.makedirs(self.main_path, exist_ok=True)
        self.tasks = {}
        self.inj_lc = []
        self.ext_lc = []
        self.schema = self.create_schema()
        # self.tab = afwTable.SourceTable.make(schema)
        self.log = {"task":[], "time":[]}
        self.calexp_data_ref = None
        self.mjds = None
        self.visits = None
        self.detectors = None
        self.htm_level = htm_level
        self.inject_table = None
        
    def add_lc(self, params, model="Pacz",  ra=None, dec=None):
        if not ra and not dec:
            if len(self.inj_lc) == 0:
                ra = random.uniform(dp0_limits[0][0], dp0_limits[0][1])
                dec = random.uniform(dp0_limits[1][0], dp0_limits[1][1])
            else:
                first_lc = self.inj_lc[0]
                f_ra = first_lc.ra; f_dec = first_lc.dec
                ra = random.uniform(f_ra-0.5, f_ra+0.5)
                dec = random.uniform(f_dec-0.5, f_dec+0.5)
        lc = LightCurve(ra, dec)
        if len(self.inj_lc) == 0:
            lc.collect_calexp(self.htm_level)
            self.calexp_data_ref = lc.calexp_data_ref
            self.mjds = lc.data["mjd"]
            self.visits =  [dataref.dataId["visit"] for dataref in self.calexp_data_ref]
            self.detectors =  [dataref.dataId["detector"] for dataref in self.calexp_data_ref]

        else:
            lc.data["mjd"] = self.mjds
        lc.simulate(params, model=model)
        self.inj_lc.append(lc)

    def inj_catalog(self, calexp):  # Estaba aca!!! Hay que ver como crear el catalogo para cada calexp segun que puntos son contenidos
        ra_values = []; dec_values  = []; src_type = []; mask=[]
        mjds = []; visits = []; detectors = []; var_mags = []
        n = len(self.mjds)
        for lc in self.inj_lc:
            if calexp.contains(lc.ra, lc.dec):
                mask.append(True)
            else:
                mask.append(False)
            var_mags.append(lc.data["mag"])
        ra_values+=[lc.ra]*sum(mask)
        dec_values+=[lc.dec]*sum(mask)
        src_type+=["Star"]*sum(mask)
        mjds.append(self.mjds[mask])
        visits.append(self.visits[mask])
        detectors.append(self.detectors[mask])
        

        inject_table = Table([np.arange(len(visits)), visits, detectors, ra_values , dec_values,
                              src_type_arr, mjds, var_mags],
                             names=['injection_id', 'visit', 'detector', 'ra', 'dec',
                                    'source_type', 'exp_midpoint', 'mag'])
        self.inject_table = inject_table

    def log_task(self, name):
        self.task_log["time"].append(time.time())
        self.task_log["task"].append(task_name)

    def inject_task(self):
        inject_config = VisitInjectConfig()
        self.tasks["Injection"]=VisitInjectTask(config=inject_config)

    def create_schema(self):
        schema = afwTable.SourceTable.makeMinimalSchema()
        schema.addField("coord_raErr", type="F", doc="Error in RA coordinate")
        schema.addField("coord_decErr", type="F", doc="Error in Dec coordinate")
        return schema

    def detection_task(self, threshold = 5, threshold_type = "stdev"):
        config = SourceDetectionTask.ConfigClass()
        config.thresholdValue = threshold
        config.thresholdType = threshold_type
        self.tasks["Detection"]=SourceDetectionTask(schema=self.schema, config=config)

    def measure_task(self, plugins=None, name = "Measurement"):
        '''plugin example = "base_SkyCoord", "base_PsfFlux", 'base_SdssCentroid' (just excecute these plugin measurment)'''
        config = SingleFrameMeasurementTask.ConfigClass()
        algMetadata = dafBase.PropertyList()
        if plugin:
            for plugin_name in config_coord.plugins.keys():
                config.plugins[plugin_name].doMeasure = False
            for plugin in plugins:
                config.plugins[plugin].doMeasure = True
        self.tasks[name]= SingleFrameMeasurementTask(schema=self.schema,config=config)

    # def characterize_task(self, psf_iter = 1): 
    #     self.name = "Characterize"
    #     config = CharacterizeImageTask.ConfigClass()
    #     config.psfIterations = psf_iter
    #     self.task = CharacterizeImageTask(config=config)

    # def deblend_task(self):
    #     self.name = "Deblend"
    #     self.task = SourceDeblendTask(schema=self.schema)

    def forced_measure_task(self, ):
        schema = self.schema #afwTable.SourceTable.makeMinimalSchema()
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
        config.plugins.names = ['base_SdssCentroid', "base_TransformedCentroid",
            "base_PsfFlux",
            "base_TransformedShape",
        ]
        config.doReplaceWithNoise = False
        self.tasks["ForcedMeasurement"] = ForcedMeasurementTask(schema, config=config)
        for j, contained in enumerate(ok):
            if contained:
                rai, deci = locs[j]
                sourceRec = forcedSource.addNew()
                coord = geom.SpherePoint(np.radians(rai) * geom.radians,
                                         np.radians(deci) * geom.radians)
                sourceRec.setCoord(coord)
                sourceRec[x_key] = pix[j][0]
                sourceRec[y_key] = pix[j][1]
            # sourceRec[type_key] = 1
        # print(forcedSource.asAstropy()[["coord_ra", "coord_dec", "centroid_x", "centroid_y"]])
        forcedMeasCat = forcedMeasurementTask.generateMeasCat(calexp, forcedSource, calexp.getWcs())
        forcedMeasurementTask.run(forcedMeasCat, calexp, forcedSource, calexp.getWcs())
        sources = forcedMeasCat.asAstropy()


    def run(self, name, **kwargs):
        if name in self.tasks and hasattr(self.tasks[name], "run"):
            result = self.tasks[name].run(**kwargs)
            self.log_task(task_name)
            return result
        else:
            print(f"Task '{name}' isn't configured or it hasn't 'run' method.")
        return None

    def time_analysis(self):
        times = self.task_log["time"]
        task_names = self.task_log["task"]
        duration = [j - i for i, j in zip(times[:-1], times[1:])]
        unique_tasks = sorted(set(task_names))
        cmap = plt.get_cmap("tab20")  
        col_task = {task: cmap(i / len(unique_tasks)) for i, task in enumerate(unique_tasks)}
        task_colors = [col_task[task] for task in task_names[:-1]]
        
        plt.figure(figsize=(27, 6))
        plt.xlim(0, 720)
        plt.ylim(0, 15)
        plt.bar(range(len(duration)), duration, color=task_colors, width=2)
        for task in unique_tasks:
            plt.bar(0, 0, color=col_task[task], label=task)
        plt.xlabel("Step")
        plt.ylabel("Duration (seconds)")
        plt.legend(title=f'Tasks - Duration: {np.sum(duration) / 60:.2f} min', ncol=6, loc=(0, 1.01))
        plt.savefig(f'{self.main_path}time_analysis-points{n_points}-HTMlevel{level}.png', bbox_inches='tight')
        print("Time analysis saved.")