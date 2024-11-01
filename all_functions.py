# Standard library imports
import os
import sys
import time
import gc
import zipfile
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import imageio
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# LSST-related imports
from lsst.rsp import get_tap_service
from lsst.daf.butler import Butler
from lsst.daf.butler.registry import ConflictingDefinitionError
import lsst.daf.base as dafBase
import lsst.afw.display as afwDisplay
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.geom as geom
import lsst.sphgeom

# LSST source injection imports
from lsst.source.injection import (
    ingest_injection_catalog, generate_injection_catalog,
    VisitInjectConfig, VisitInjectTask
)

# LSST pipe tasks imports
from lsst.pipe.tasks.registerImage import RegisterConfig, RegisterTask
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask

# LSST measurement imports
from lsst.meas.algorithms.detection import SourceDetectionTask
from lsst.meas.deblender import SourceDeblendTask
from lsst.meas.base import SingleFrameMeasurementTask, ForcedMeasurementTask


# afwDisplay.setDefaultBackend('matplotlib')
# plt.style.use('tableau-colorblind10')

butler_config = 'dp02'
collections = '2.2i/runs/DP0.2'
butler = Butler(butler_config, collections=collections)

def calculate_htm_id(ra, dec, level=20):
    pixelization = lsst.sphgeom.HtmPixelization(level)
    htm_id = pixelization.index(
        lsst.sphgeom.UnitVector3d(
            lsst.sphgeom.LonLat.fromDegrees(ra, dec)
        )
    )
    circle = pixelization.triangle(htm_id).getBoundingCircle()
    scale = circle.getOpeningAngle().asDegrees()*3600.
    level = pixelization.getLevel()
    print("{:<20}".format(f'({ra}, {dec})') + f'HTM ID={htm_id} at level={level} is bounded by a circle of radius ~{scale:0.2f} arcsec.')

    return htm_id
    
def calexp_from_loc(ra, dec, band = "i", level = 20):
    htm_id = calculate_htm_id(ra, dec, level)
    datasetRefs = butler.registry.queryDatasets("calexp", htm20=htm_id,
                                                where=f"band = '{band}'")
    datasetRefs_list = [ref for ref in datasetRefs]
        # Example: ref = DatasetRef(DatasetType('calexp', {band, instrument, detector, physical_filter, visit_system, visit}, ExposureF), 
        # {instrument: 'LSSTCam-imSim', detector: 106, visit: 1231987, band: 'i', 
        # physical_filter: 'i_sim_1.4', visit_system: 1}, run='2.2i/runs/DP0.2/v23_0_0_rc5/PREOPS-905/20220104T085126Z', 
        # id=6894509b-e233-484e-b04d-416a5805c350) 
    
    print("{:<20}".format("") + f"Found {len(list(datasetRefs))} calexps")

    # From that images, save the exposure times by searching them per visit and detector in ccdVisitTable
    ccd_visit = butler.get('ccdVisitTable') # I guess this butler have an specific OpSim (observation strategy)
    exp_midpoints = []
    visits = []
    detectors = []
    
    for d in datasetRefs_list:
        did = d.dataId
        ccdrow = (ccd_visit['visitId'] == did['visit']) & (ccd_visit['detector'] == did['detector'])
        exp_midpoints.append(ccd_visit[ccdrow]['expMidptMJD'].values[0])
        visits.append(did['visit'])
        detectors.append(did['detector'])
    
    exp_midpoints = np.array(exp_midpoints)
    visits = np.array(visits)
    detectors = np.array(detectors) 
    return exp_midpoints, visits, detectors, datasetRefs_list
    
import numpy as np
import matplotlib.pyplot as plt

def uLens_Pacz(t, t_0, t_E, u_0, m_base, plot=False):
    """
    Calculate the microlensing magnification using the Paczynski formula.
    
    Parameters:
    t (float or np.ndarray): Time(s) at which to calculate the magnification.
    t_0 (float): Time of closest approach.
    t_E (float): Einstein timescale.
    u_0 (float): Minimum impact parameter.
    m_base (float): Baseline magnitude of the source star.
    plot (bool): If True, plot the magnification curve.
    
    Returns:
    float or np.ndarray: Magnitude(s) at the given time(s).
    """
    u_t = np.sqrt(u_0**2 + ((t - t_0) / t_E)**2)
    A_t = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))
    
    # Convert magnification to magnitude
    m_t = m_base - 2.5 * np.log10(A_t)
    
    if plot:
        t_plot = np.linspace(np.min(t), np.max(t), 1000)
        A_plot = (np.sqrt(u_0**2 + ((t_plot - t_0) / t_E)**2)**2 + 2) / \
                 (np.sqrt(u_0**2 + ((t_plot - t_0) / t_E)**2) * np.sqrt(np.sqrt(u_0**2 + ((t_plot - t_0) / t_E)**2)**2 + 4))
        m_plot = m_base - 2.5 * np.log10(A_plot)
        
        plt.plot(t_plot, m_plot, color='gray')
        plt.scatter(t, m_t, color='red')
        plt.xlabel('Time')
        plt.ylabel('Magnitude')
        plt.title('Microlensing - Paczynski')
        plt.gca().invert_yaxis()  # Invert y-axis to show magnitude properly
        plt.show()
    
    return m_t

def warp_img(ref_img, img_to_warp, ref_wcs):
    '''Warp an image to the same orientation as a reference image

    Parameters
    ----------
    ref_img: `ExposureF`
        Reference image to warp to
    img_to_warp: `ExposureF`
        Image to warp to the reference orientation
    ref_wcs: `WCS` object
        WCS of the reference image
    wcs_to_warp: `WCS` object
        WCS of the input image to be warped
    '''
    wcs_to_warp = img_to_warp.getWcs()
    config = RegisterConfig()
    task = RegisterTask(name="register", config=config)
    warpedExp = task.warpExposure(img_to_warp, wcs_to_warp, ref_wcs,
                                  ref_img.getBBox())

    return warpedExp

def create_catalog(var_mags, exp_midpoints, visits, detectors, ra, dec):
    ra_arr = np.full((len(var_mags)), ra)
    dec_arr = np.full((len(var_mags)), dec)
    id_arr = np.arange(0, len(var_mags), 1)
    src_type_arr = np.full((len(var_mags)), 'Star')
    
    inject_table = Table([id_arr, visits, detectors, ra_arr, dec_arr,
                          src_type_arr, exp_midpoints, var_mags],
                         names=['injection_id', 'visit', 'detector', 'ra', 'dec',
                                'source_type', 'exp_midpoint', 'mag'])
    return inject_table

# Get reference data ID and relevant data from the reference calexp
def get_calexp_data(datasetRef):
    ref_dataId = datasetRef.dataId
    calexp_ref = butler.get('calexp', dataId=ref_dataId)
    psf_ref = calexp_ref.getPsf()
    phot_calib_ref = calexp_ref.getPhotoCalib()
    wcs_ref = calexp_ref.getWcs()
    xy_ref = wcs_ref.skyToPixel(geom.SpherePoint(ra*geom.degrees, dec*geom.degrees))  # Convert reference RA and Dec to pixel coordinates
    # x_ref = int(np.round(xy_ref.x))
    # y_ref = int(np.round(xy_ref.y))
    return calexp_ref, psf_ref, phot_calib_ref, wcs_ref, xy_ref

# Injection and cutting

def inject_catalog(datasetRefs_list, inject_table, start_ind=0, finish_ind=-1, warp=False, cut=False):
    imgs = []
    dataids = []
    mjd_mid_times = []
    mags_injected = []
 
    inject_config = VisitInjectConfig()
    inject_task = VisitInjectTask(config=inject_config)
    datasetRefs=datasetRefs_list[start_ind:finish_ind]
    for i, ds in enumerate(datasetRefs):
        print(f'{i+1}/{len(datasetRefs)}')
        calexp_i, psf_i, phot_calib_i, wcs_i, xy_i = get_calexp_data(ds)    
        try:
            injected_output_i = inject_task.run(
                injection_catalogs=[inject_table[i]],
                input_exposure=calexp_i.clone(),
                psf=psf_i,
                photo_calib=phot_calib_i,
                wcs=wcs_i,)
            injected_exposure_i = injected_output_i.output_exposure
            injected_catalog_i = injected_output_i.output_catalog
                
            mjd_mid_times.append(inject_table[i]['exp_midpoint'])
            mags_injected.append(inject_table[i]['mag'])
            dataids.append(ds.dataId)
    
            if warp:
                img = warp_img(calexp_ref, injected_exposure_i, wcs_ref, wcs_i)
                print("Warped")
            else:
                img = injected_exposure_i
            if cut != False:
                print("Cutting")
                ra = cut[0]
                dec = cut[1]
                xy = img.getWcs().skyToPixel(geom.SpherePoint(ra*geom.degrees, dec*geom.degrees))
                x = int(np.round(xy.x))
                y = int(np.round(xy.y))
                img = cutout(img, x, y, 301)
                print("Cut finished")
            imgs.append(img)
            print(len(imgs))
        except:
            # Some visits don't actually overlap the point where we're injecting a star
            print('No sources to inject for visit ', inject_table[i]['visit'])
    return dataids, mjd_mid_times, mags_injected, imgs

def Injection():
    inject_config = VisitInjectConfig()
    inject_task = VisitInjectTask(config=inject_config)
    return inject_task
    
def inject(calexp_data, inject_data, inject_task, fits=False):
    calexp, psf, phot_calib, wcs, xy = get_calexp_data(calexp_data)    
    try:
        injected_output = inject_task.run(
            injection_catalogs=[inject_data],
            input_exposure=calexp.clone(),
            psf=psf,
            photo_calib=phot_calib,
            wcs=wcs,)
        injected_exposure = injected_output.output_exposure
        if fits!= False:
            injected_exposure.writeFits(fits)
        # injected_catalog = injected_output.output_catalog
            
        mjd = inject_data['exp_midpoint']
        mag = inject_data['mag']
        dataid = calexp_data.dataId
        return dataid, mjd, mag, injected_exposure
    except Exception as e:
        # Some visits don't actually overlap the point where we're injecting a star
        print('No sources to inject for visit ', inject_data['visit'], "Error:", e)
        return None, None, None, None
    

def SkyToPix(ra,dec, img):
    xy = img.getWcs().skyToPixel(geom.SpherePoint(ra*geom.degrees, dec*geom.degrees))
    x = int(np.round(xy.x))
    y = int(np.round(xy.y))
    return x , y

from lsst.geom import SpherePoint, Angle

# def SkyToPix(ra, dec, img):
#     """ Convert RA, Dec to pixel coordinates using WCS """
#     if isinstance(ra, u.Quantity):
#         ra = Angle(ra.to_value(u.deg), unit='deg')
#     if isinstance(dec, u.Quantity):
#         dec = Angle(dec.to_value(u.deg), unit='deg')

#     sky_point = SpherePoint(ra, dec)
#     xy = img.getWcs().skyToPixel(sky_point)
    
#     x = int(np.round(xy.x))
#     y = int(np.round(xy.y))
    
#     return x, y


def PixToSky(x, y, img):
    sphere_point = img.getWcs().pixelToSky(geom.Point2D(x, y))
    ra = sphere_point.getRa().asDegrees()
    dec = sphere_point.getDec().asDegrees()
    return ra, dec

def cutout(img, ra, dec , size):
    '''Create a cutout of an input image array

    Parameters
    ----------
    im: `Image`
        Input image (extracted from an ExposureF) to cut down
    xcen, ycen: `int`
        Integer XY coordinates to center the cutout on
    size: `int`
        Width in pixels of the resulting image
    '''
    x,y=SkyToPix(ra, dec, img)
    try:
        return img[x-size/2:x+size/2, y-size/2:y+size/2]
    except Exception as e:
        print(f"Couldn't cut image. Expeption: {e}")
        print(x,y)
        return img

import matplotlib.patches as patches

def calexp_plot(img, title, fig=None, ax=None, warp=None, cut=None, point_out=False, cutsize=None, col=None):
    '''warp: calexp_ref'''
    if warp!=None:
        img = warp_img(warp, img, warp.getWcs())
    if type(cut)!=type(None):
        ra, dec = cut[:,0], cut[:,1]
        if locs[ok].shape!=(0,2):
            size = 401
            if cutsize!=None:
                size = cutsize
            img = cutout(img, ra[0], dec[0], size)
        else:
            print(f"Couldn't cut image {title}")
    if fig==None:
        fig, ax = plt.subplots(1)    
    display0 = afwDisplay.Display(frame=fig)
    # display0.scale('linear', 'zscale')
    display0.scale('linear', min=-100, max=250)
    display0.mtv(img.image)
    plt.title(title, fontsize=8)
    if col!=None:
        col=colors
    if point_out:
        for h, (rai, deci) in enumerate(zip(ra,dec)):
            if img.containsSkyCoords(np.array(rai) * u.deg,np.array(deci) * u.deg)[0]:
                x,y=SkyToPix(rai, deci, img)
                rect = patches.Rectangle((x - size/10, y - size/10), size/5, size/5, edgecolor=col[h], linewidth=3, facecolor='none')
                ax.add_patch(rect)
    ax.axis('off')

def flux_to_mag(Flux, Flux_err = "", F0=27.85):
    '''Use numpy arrays, no list'''
    if type(Flux_err) != str:
        top = Flux + Flux_err
        bot = Flux - Flux_err
        top_mag = F0 - 2.5 * np.log10(np.abs(top))
        try:
            bot_mag = F0 - 2.5 * np.log10(np.abs(bot))
        except RuntimeWarning as e:
            print(e, Flux, Flux_err)
        return F0 - 2.5 * np.log10(np.abs(Flux)), np.abs(top_mag - bot_mag)
    else:
        return F0 - 2.5 * np.log10(np.abs(Flux))

def initialize_tasks():
    schema = afwTable.SourceTable.makeMinimalSchema()
    algMetadata = dafBase.PropertyList()
    # Measurment
    config = SingleFrameMeasurementTask.ConfigClass()
    measure_task = SingleFrameMeasurementTask(schema=schema,
                                                       config=config,
                                                       algMetadata=algMetadata)
    # Characterize
    config = CharacterizeImageTask.ConfigClass()
    config.psfIterations = 1
    char_image_task = CharacterizeImageTask(config=config)

    # Source Detection 
    config = SourceDetectionTask.ConfigClass()
    config.thresholdValue = 5
    config.thresholdType = "stdev"
    detect_task = SourceDetectionTask(schema=schema, config=config)

    # Source Deblending
    deblend_task = SourceDeblendTask(schema=schema)
    tab = afwTable.SourceTable.make(schema)
    return measure_task, char_image_task, detect_task, deblend_task, schema, tab

def generate_light_curves(exp_midpoints, model):
    if model["name"]=="Pacz":
        t_0, t_E, u_0, m_t = model["t_0"], model["t_E"], model["u_0"], model["m_t"],
        var_mags = uLens_Pacz(exp_midpoints, t_0, t_E, u_0, m_t, plot=False)
        t = np.linspace(min(exp_midpoints), max(exp_midpoints), 1000)
        ideal = uLens_Pacz(t, t_0, t_E, u_0, m_t, plot=False)
    return var_mags, ideal, t

def detect_sources(tab, injected_exposure, detect_task, deblend=None, measure=None):
    detect_result = detect_task.run(tab, injected_exposure)
    sources = detect_result.sources
    # sources = sources.asAstropy()
    if deblend != None:
        deblend.run(injected_exposure, sources)
    if measure != None:
        measure.run(measCat=sources, exposure=injected_exposure)
    return sources

def source_photometry(sources, ra, dec, injected_exposure, delta = 0.00005):
    # sources = sources.copy(True)
    # sources = sources.asAstropy()
    # source = sources[((abs(sources["coord_ra"]/np.pi*180)-ra) < delta)&(abs(sources["coord_dec"]/np.pi*180-dec) < delta)]
    # n=len(source)
    for source in sources:
        if (abs(float(source["coord_ra"])/np.pi*180-ra) < delta)  and (abs(float(source["coord_dec"])/np.pi*180-dec) < delta):
            n = len(sources[(abs(sources["coord_ra"]/np.pi*180-ra) < delta) & (abs(sources["coord_dec"]/np.pi*180-dec) < delta)])
            if n > 1:
                print(f"Warning: {n} sources close ({delta}) to ra,dec = {ra},{dec}")
                # Calculate distances to the input ra, dec
                distances = np.sqrt((float(source["coord_ra"]) / np.pi * 180 - ra)**2 + 
                                    (float(source["coord_dec"]) / np.pi * 180 - dec)**2)
                # Select the source with the minimum distance
                closest_source_idx = np.argmin(distances)
                source = source[closest_source_idx:closest_source_idx+1]

    flux = source["base_PsfFlux_instFlux"]
    flux_err = source["base_PsfFlux_instFluxErr"]
    print(f"Flux computed for ra,dec = {source['coord_ra']},{source['coord_dec']}")
    return flux, flux_err
    if n==0:
        print(f"Warning: No source close ({delta}) ra,dec = {ra},{dec}")
        return None, None
    
    
def toMag(injected_exposure, flux, flux_err):
    photoCalib = injected_exposure.getPhotoCalib()
    measure = photoCalib.instFluxToMagnitude(flux, flux_err)
    mag = measure.value
    mag_err = measure.error
    # flux = source.get('base_CircularApertureFlux_3_0_instFlux')
    # flux_err = source.get('base_CircularApertureFlux_3_0_instFluxErr')
    # zf = photoCalib.getInstFluxAtZeroMagnitude() # zeroflux → pc.instFluxToMagnitude(zf)~0
    # mag, mag_err = flux_to_mag(flux, flux_err, -2.5 * np.log10(zf))
    return mag, mag_err

def lc_plot(mjds, mags_measured, mags_errors, mags_injected=None, t=None, ideal=None, model=None, ax=None, fig=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    if mags_injected is not None:
        ax.plot(mjds, mags_injected, 'bo', label="Injected", markersize=5)
    ax.errorbar(mjds, mags_measured, yerr=mags_errors, fmt='ro', label="Measured", markersize=4)
    
    if t is not None and mags_injected is not None:
        t_aux = t[t <= mjds[-1]]
        ax.plot(t_aux, ideal[:len(t_aux)], "gray", label="Pacz")
    
    ax.set_xlabel('expMid')
    ax.set_ylabel('mag')
    title = ""
    for i in model:
        title+=i+": "+str(model[i])+" | "
    ax.set_title(title, fontsize=10)
    ax.invert_yaxis()
    ax.legend()
    plt.close()
    
    return fig, ax
    # ax2.set_title(f'{i}. visit: {dataid["visit"]}, expMid: {mjd:0.5F}, mag={mag:0.2F}', fontsize=8)
    # x,y = SkyToPix(ra, dec, img)
    # rect = patches.Rectangle((x - 20, y - 20), 40, 40, linewidth=1, edgecolor='r', facecolor='none')
    # ax2.add_patch(rect)

def make_gif(gif_name, filenames, duration = 500):
    with imageio.get_writer(gif_name, mode='I', duration=duration) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            from shapely.geometry import Point, Polygon



def get_calexp_corners(calexp):
    x0 = float(calexp.getX0())
    y0 = float(calexp.getY0())
    width = calexp.getWidth()
    height = calexp.getHeight()
    xcorners = [x0, x0+width, x0+width, x0]
    ycorners = [y0, y0, y0+width, y0+width] 
    wcs=calexp.getWcs()
    ra_corners = []
    dec_corners = []
    for i in range(len(xcorners)):
        radec = wcs.pixelToSky(xcorners[i], ycorners[i])
        ra_corners.append(radec.getRa().asDegrees())
        dec_corners.append(radec.getDec().asDegrees())
    return ra_corners, dec_corners
    
def isin_calexsp(ra, dec, RA, DEC, count=True):
    point = Point(ra, dec)
    if count:
        j=0
        for ra_corners, dec_corners in zip(RA, DEC):
            polygon = Polygon(zip(ra_corners, dec_corners))
            if polygon.contains(point):
               j+=1
        return j
    else:
        for ra_corners, dec_corners in zip(RA, DEC):
            polygon = Polygon(zip(ra_corners, dec_corners))
            if not polygon.contains(point):
               return False
        return True
        
def get_common_area(RA, DEC):
    polygons = [Polygon(zip(ra_corners, dec_corners)) for ra_corners, dec_corners in zip(RA, DEC)]
    common_area = polygons[0]
    for poly in polygons[1:]:
        common_area = common_area.intersection(poly)
    return common_area


def inj_ext_calexp(calexp_data, inject_data, inject_task, detect_task, tab, deblend=None, measure=None, fits=None):
    # Inject
    dataid, mjd, mag_inj, injected_exposure = inject(calexp_data, inject_data, inject_task, fits=fits)
    if (dataid, mjd, mag_inj, injected_exposure) == (None, None, None, None):
        return None
    # Extraction
    sources = detect_sources(tab, injected_exposure, detect_task, deblend=deblend, measure=measure)
    flux, flux_err = source_photometry(sources, inject_data["ra"], inject_data["dec"], injected_exposure, delta = 0.0005)
    mag, mag_err = toMag(injected_exposure, flux, flux_err) if (flux, flux_err)!= (None, None) else (None, None)
    return dataid, mjd, mag_inj, injected_exposure, flux, flux_err, mag, mag_err


def inj_ext_lc(band, ra, dec, model, start=0, finish=None, htm_lvl= 20, csv = None, frames = False, gif = None, fits = None):
    exp_midpoints, visits, detectors, datasetRefs_list = calexp_from_loc(ra, dec, band, level = htm_lvl)
    calexp_ref, psf_ref, phot_calib_ref, wcs_ref, xy_ref= get_calexp_data(datasetRefs_list[start])

    finish = len(datasetRefs_list) if finish==None else finish
    
    mags_measured = []; mags_injected = []; mags_errors = []; detector = []; visit = []
    fluxs = []; flux_errs = []; dataids = []; mjds = []; filenames = []; imgs = []

    var_mags, ideal, mjd_ideal = generate_light_curves(exp_midpoints, model) 
    inject_table = create_catalog(var_mags, exp_midpoints, visits, detectors)
    
    measure_task, char_image_task, detect_task, deblend_task, schema, tab = initialize_tasks()
    inject_task=Injection() 
    
    for i, (calexp_data, inject_data) in enumerate(zip(datasetRefs_list[start:finish], inject_table[start:finish])):
        print(f"{'-'*15} Injection   {i+1}/{finish-start} {'-'*15}")
        if fits == True:
            name_fits = f"calexp{i}-m{m_t}.fits"
        else:
            name_fits = None
        result = inj_ext_calexp(calexp_data, inject_data, inject_task, detect_task, tab, deblend=deblend_task, measure=measure_task, fits=name_fits)
        if result == None:
            continue
        else:
            dataid, mjd, mag_inj, injected_exposure, flux, flux_err, mag, mag_err = result
                                
        mjds.append(mjd)
        mags_measured.append(mag)
        mags_errors.append(mag_err)
        fluxs.append(flux)
        flux_errs.append(flux_err)
        dataids.append(dataid)
        mags_injected.append(mag_inj)
        detector.append(dataid["detector"])
        visit.append(dataid["visit"])
        
        if frames:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            calexp_plot(injected_exposure, f"expMid: {mjd} - mag_inj={mag_inj}",fig, ax2, warp=calexp_ref) # arreglar = Acá hay algo None_type a veces, 
            lc_plot(mjds, mags_injected, mags_measured, mags_errors, mjd_ideal, ideal, model, ax=ax1) 
            plt.tight_layout()
            filename = f"frame{i}-m{m_t}.png"
            plt.savefig(filename, bbox_inches='tight')
            filenames.append(filename)
    
            

        print(f"MJD: {mjd} | Injeted: {mag_inj} | Measured: {mag} | Error: {mag_err}")
        print(f"                            Flux: Measured: {flux} | Error: {flux_err}")
        del injected_exposure
        
    data = {'mags_measured': mags_measured,'mags_injected': mags_injected,'mags_errors': mags_errors,'fluxs': fluxs,'flux_errs': flux_errs,'detector': detector,'visit': visit,'mjds': mjds}
    
    if csv!=None:
        df = pd.DataFrame(data)
        df.to_csv(csv, index=False)
    if frames and (gif!=None):
        make_gif(gif, filenames)
    return data


def memory_var():
    gc.collect()
    global_vars = globals()
    local_vars = locals()
    user_defined_vars = {**global_vars, **local_vars}
    user_defined_vars = {k: v for k, v in user_defined_vars.items() if not (k.startswith('__') or isinstance(v, (type(sys), type(gc))))}
    size  = [(k, round(sys.getsizeof(v) / (1024 ** 2),5)) for k, v in user_defined_vars.items()]
    return objetos_con_tamano

def check_edge( calexp,ra, dec):
    x, y = SkyToPix(ra, dec, calexp)
    edge =True
    for bx in [calexp.getBBox().minX,calexp.getBBox().maxX]:
        if abs(bx-x)<50:
            edge=False
            break
    for by in [calexp.getBBox().minY,calexp.getBBox().maxY]:
        if abs(by-y)<50:
            edge=False
            break
    return edge

from astropy.wcs import WCS
import matplotlib.patches as patches


import astropy.units as u
from matplotlib.ticker import FuncFormatter
def get_calexp_corners(calexp):
    x0 = float(calexp.getX0())
    y0 = float(calexp.getY0())
    width = calexp.getWidth()
    height = calexp.getHeight()
    xcorners = [x0, x0+width, x0+width, x0]
    ycorners = [y0, y0, y0+width, y0+width] 
    wcs=calexp.getWcs()
    ra_corners = []
    dec_corners = []
    for i in range(len(xcorners)):
        radec = wcs.pixelToSky(xcorners[i], ycorners[i])
        ra_corners.append(radec.getRa().asDegrees())
        dec_corners.append(radec.getDec().asDegrees())
    return ra_corners, dec_corners
    
def isin_calexsp(ra, dec, RA, DEC, count=True):
    point = Point(ra, dec)
    if count:
        j=0
        for ra_corners, dec_corners in zip(RA, DEC):
            polygon = Polygon(zip(ra_corners, dec_corners))
            if polygon.contains(point):
               j+=1
        return j
    else:
        for ra_corners, dec_corners in zip(RA, DEC):
            polygon = Polygon(zip(ra_corners, dec_corners))
            if not polygon.contains(point):
               return False
        return True
def calexp_plot(img, title, fig=None, ax=None, warp=None, cut=None, point_out=False, cutsize=None, col=None):
    '''warp: calexp_ref'''
    if warp is not None:
        img = warp_img(warp, img, warp.getWcs())

    if cut is not None:
        ra, dec = cut[:,0], cut[:,1]
        if ra.shape != (0,) and dec.shape != (0,):
            size = 401
            if cutsize is not None:
                size = cutsize
            img = cutout(img, ra[0], dec[0], size)
        else:
            print(f"Couldn't cut image {title}")

    if fig is None:
        fig = plt.figure()
    if ax is None:
        wcs_projection = WCS(img.getWcs().getFitsMetadata())
        ax = fig.add_subplot(122, projection=wcs_projection)

    ax.set_title(title, fontsize=8)
    ax.set_xlabel('ra (degrees)',fontsize=8)
    ax.set_ylabel('dec (degrees)',fontsize=8)


    # Use WCS projection to set axis labels to RA/Dec in degrees
    ax.coords['ra'].set_axislabel('RA (degrees)', fontsize=8)
    ax.coords['dec'].set_axislabel('Dec (degrees)', fontsize=8)

    # Ensure RA is displayed in degrees, not hours
    ax.coords['ra'].set_major_formatter('d.ddd')
    ax.coords['dec'].set_major_formatter('d.ddd')
    
    # Set the number of ticks or spacing between ticks
    ax.coords['ra'].set_ticks(spacing=0.004 * u.deg)  # Adjust spacing as needed
    ax.coords['dec'].set_ticks(spacing=0.003 * u.deg)  # Adjust spacing as needed

    ax.coords['ra'].set_ticklabel(fontsize=6)
    ax.coords['dec'].set_ticklabel(rotation=-30, fontsize=6)
    ax.coords['ra'].set_ticks_position('b')  # 'b' stands for bottom
    ax.coords['ra'].set_ticklabel_position('b')  # Ensure tick labels are on the bottom
    # ax.set_position([ax.get_position().x0, ax.get_position().y0 - 1, ax.get_position().width, ax.get_position().height])

    ax.imshow(img.image.array, cmap='gray', vmin=-200.0, vmax=400,
              extent=(img.getBBox().beginX, img.getBBox().endX,
                      img.getBBox().beginY, img.getBBox().endY),
              origin='lower')
    ax.grid(color='white', ls='--', lw=0.2)

    if point_out:
        for h, (rai, deci) in enumerate(zip(ra, dec)):
            rai_deg = rai * u.deg if not isinstance(rai, u.Quantity) else rai
            deci_deg = deci * u.deg if not isinstance(deci, u.Quantity) else deci
            
            if img.containsSkyCoords(rai_deg, deci_deg)[0]:
                x, y = SkyToPix(rai, deci, img)
                rect = patches.Rectangle((x - size/20, y - size/20), size/10, size/10, 
                                         edgecolor=col[h] if col else 'red', linewidth=2, facecolor='none')
                ax.add_patch(rect)

def run_name():
    date = datetime.now()
    nombre_corrida = date.strftime("%Y%m%d_%H%M%S")
    return nombre_corrida

