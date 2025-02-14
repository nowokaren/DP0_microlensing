from light_curves import LightCurve
from exposures import Calexp
from task import Run
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm.notebook import tqdm
from memory_profiler import profile

radius = 0.2          #  [0.1,0.2,0.3, 0.5]
density = 50/0.05     #  ]np.array([10,50, 100, 500])/0.05
n_max_calexps = 1000
bands = "ugrizy"
problems = []
delta_mag = np.array([2.5,2,1.5,1.0,0.5,0])[-len(bands):]
# print(delta_mag=)


process = Run(ra=57.59451632893858, dec=-32.481152201226145, 
              density = density, scale = radius, bands = bands, 
              name = f"run_{density:.0f}dens_{int(radius*10):02d}rad_{bands}_{n_max_calexps}calexps-test", 
              calexps_method="overlap", measure_method ="ForcedMeas", #)
              data_events="runs/run_1000dens_02rad_ugrizy_1000calexps/data_events.csv")

print(process)
process.inject_task()
schema = process.measure_task()

process.load_calexp(n_max_calexps)
process.data_calexps.to_csv(process.main_path+'/data_calexp.csv', index=False)

for i, (j, event) in enumerate(tqdm(process.data_events.iterrows(), desc="Loading events:")):
    lc_ra, lc_dec = event["ra"], event["dec"]
    params = {"t_0": event["t_0"],
           "t_E": event["t_E"], 
           "u_0": event["u_0"]}
    params["m_base"]=event["m_base"]
    process.add_lc(lc_ra, lc_dec, params, event_id=event["event_id"], band=event["band"])
process.data_events.to_csv(process.main_path+'/data_events.csv', index=False) 

# print(process)
# process.inject_task()
# schema = process.measure_task()

# process.collect_calexp(n_max_calexps)
# process.data_calexps.to_csv(process.main_path+'/data_calexps.csv', index=False)

# for event_id, m in enumerate(tqdm(np.linspace(17,22,process.n_lc), desc="Loading events")):
#     lc_ra, lc_dec = process.generate_location()
#     params = {"t_0": random.uniform(60300,61500),
#    "t_E": random.uniform(20, 200), 
#    "u_0": random.uniform(0.1,1)}
#     for band, dm in zip(process.bands, delta_mag):
#         params["m_base"]=m+dm
#         process.add_lc(lc_ra, lc_dec, params, event_id=event_id, band=band)
# process.data_events.to_csv(process.main_path+'/data_events.csv', index=False) 

process.sky_map()
process.log_task("Add and simulate light curves")

@profile
def process_calexps(process, n_max_calexps):
    for idx, data in process.data_calexps.iterrows():
        if idx == n_max_calexps:
            break
        print(f" ------ CALEXP {idx+1}/{n_max_calexps} ------")
        data_id = data[['detector', 'visit']].to_dict(); print(f"{data_id=}")
        band = data["band"]; print("Band: ", band)
        calexp = Calexp(data_id)
        if calexp.overlaps(process.region):
            process.add_data_calexp(data_id, "overlap", True)
            injection_catalog = process.create_injection_table(calexp, band)
            injected_exposure, injected_catalog = process.inject_calexp(calexp, injection_catalog, save_fit=f"calexp_{idx}_{band}.fit")
            if injected_catalog is not None:
                calexp_lcs = "-"+"-".join(list(map(str, injected_catalog["lc_id"].value)))+"-"
                process.add_data_calexp(data_id, "lc_ids", calexp_lcs)
                calexp_inj = Calexp(injected_exposure)
                table, sources = process.measure_calexp(schema, calexp_inj, injected_catalog)
                for _, source in table.iterrows():
                    lc = process.inj_lc[source["lc_id"]]
                    process.plot_FoV(lc, calexp_inj, show=False)
                    flux, flux_err, mag, mag_err = source[["flux", "flux_err", "mag", "mag_err"]]
                    lc.add_data(data_id, [flux, flux_err, mag, mag_err])
                del calexp_inj, table, sources, injected_catalog, injection_catalog
            else:
                print("Injection catalog is empty. Skipping...")
        else:
            print("No intersection between calexp and region. Skipping...")
        del calexp
    process.save_data()
    print("Injected points per light curve", np.array(process.data_events["points"].values, dtype=int) )
# process.sky_map(calexps=n_max_calexps)
# for band in process.bands:
#     process.sky_map(calexps=n_max_calexps, band=band)


if __name__ == "__main__":
    process_calexps(process, n_max_calexps)