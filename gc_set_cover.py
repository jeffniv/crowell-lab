#gc_set_cover.py // Functions used for set cover problem applied to GeoCARB mission
#@author: Jeff Nivitanont, University of Oklahoma, 2018/03/12

import numpy
import datetime as dt
import time
import calendar
from numba import jit
from shapely.ops import transform
from functools import partial
import pyproj
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from pylab import *
import winsound
import geopandas as gpd

geo_lon = -85.    #satellite lon
geo_lat = 0.      #satellite lat
geo_ht = 42.336e6 #fixed satellite height
surf_ht = 6.370e6 #uniform surface height
sol_ht = 149.6e9 #fixed solar height
r_earth = 6.371e6 # radius of Earth
north_solst = 354 # date of southern summer solstice
declin = 23.44
earth_sun = 149.6e9 #distance to sun
atrain_alt = 705e3 
dtor = pi/180.   #convert degs to rads

#for converting lat/lon to Geo
geo = ccrs.Geostationary(central_longitude=-85.0, satellite_height=42.336e6)
geo_proj4 = geo.proj4_init
latlon = {'init' :'epsg:4326'}

#partially applied function from pyproj for converting from lat/lon to geo
project = partial( pyproj.transform, pyproj.Proj(latlon), pyproj.Proj(geo_proj4) )
#partially applied function from pyproj for converting from lat/lon to geo
project2 = partial( pyproj.transform, pyproj.Proj(geo_proj4), pyproj.Proj(latlon))


def beep():
    '''it goes beep x 4'''
    duration = 150  # millisecond
    freq = 880  # Hz
    for i in range(4):
        winsound.Beep(freq, duration)
        
#from earth_calcs.py
@jit
def scalar_earth_angle( lat1, lon1, lat2, lon2):
    """ angle on great circle between two points """
    theta1 = lat1 *dtor
    phi1 = lon1 *dtor
    theta2 = lat2 * dtor
    phi2 = lon2 * dtor
    p1 = numpy.vstack((cos(theta1)*cos(phi1),cos(theta1)*sin(phi1),sin( theta1))).T
    p2 = numpy.vstack((cos(theta2)*cos(phi2), cos( theta2)* sin( phi2), sin( theta2))).T
    dsq = ((p1-p2)**2).sum(-1)
    return numpy.arccos((2 -dsq)/2.)/dtor

#from earth_calcs.py
@jit
def scalar_subsol( day):
    "subsolar lat-lon given decimal day-of-year "
    lat = -declin * numpy.cos(2*pi* numpy.mod(365 + day - north_solst,  365.)/365.)
    lon = numpy.mod( 180. -360.*(day -numpy.floor(day)), 360.)
    return lat, lon

#from earth_calcs.py
@jit
def zenith_angle_cosine_batch( viewer,target):
    """ gives the zenith angle of a target  (r theta, phi) from the viewer (r, theta phi), theta, phi and result are in degrees"""
    centre_angle = scalar_earth_angle( viewer[:,1],viewer[:,2], target[:,1], target[:,2]) # angle between the two locations at centre of earth
    dist = (viewer[:,0]**2 + target[:,0]**2 -2.*target[:,0]*viewer[:,0]*numpy.cos( centre_angle*dtor))**0.5 # cosine rule
    cos_zenith = -0.5*(dist**2+viewer[:,0]**2-target[:,0]**2)/(dist*viewer[:,0])  # the minus makes it a zenith angle
    return numpy.arccos(cos_zenith)/dtor

@vectorize #this is a NumPy decorator
def airmass(lat, lon, time):
    '''gives the airmass at a lat,lon point of time'''
    start_day = dt.datetime(time.year,1,1,0,0,0)
    view_lat = lat #viewer on ground
    view_lon = lon
    day_dec = (time-start_day).total_seconds()/86400.  #decimal julian date for scalar_subsol func
    subsol_lat,subsol_lon = scalar_subsol(day_dec) #sub-solar lat/lon
    
    sol = array([array([sol_ht,subsol_lat,subsol_lon])])
    ex = array([array([surf_ht,view_lat,view_lon])])
    sol_zenith = zenith_angle_cosine_batch(ex,sol) #(viewer, target)

    sat = array([array([geo_ht,0.,geo_lon])])
    sat_zenith = zenith_angle_cosine_batch(ex,sat)

    return 1./cos(dtor*sol_zenith) + 1./cos(dtor*sat_zenith)

@jit #using jit and numpy.vectorized airmass func
def calc_afmesh_5min(xcoords, ycoords, month):
    '''calculates an array of lat, lon points in a time interval and returns a 3-D array, [time, lat, lon]'''
    time_intv = list([dt.datetime(2018, month, 15, 12)])
    for i in arange(1,288):
        time_intv.append(time_intv[i-1]+dt.timedelta(0,300))
    day_mesh = zeros([len(time_intv), len(ycoords), len(xcoords) ])
    for i in range(len(time_intv)):
        day_mesh[i,:,:] = airmass(lat=ycoords, lon=xcoords, time = time_intv[i])
    day_mesh[day_mesh < 2.0] = float(inf)
    return day_mesh

@jit #using jit and numpy.vectorized airmass func
def calc_af_min_mesh(xcoords, ycoords, time_intv):
    '''calculates an array of lat, lon points in a time interval and returns the minimum values
    in a 2-D array, [lat, lon]'''
    intv = time_intv
    min_mesh = airmass(lat=ycoords, lon=xcoords, time = intv[0])
    min_mesh[min_mesh < 2.0] = float(inf)
    for i in arange(1,len(intv)):
        temp_mesh = airmass(lat=ycoords, lon=xcoords, time = intv[i])
        temp_mesh[temp_mesh < 2.0] = float(inf)
        min_mesh = minimum(temp_mesh, min_mesh)
    return min_mesh


@jit
def calc_block_weight(block, universe, mesh, weights, covered, last):
    '''
    Weight function for Greedy Algorithm
    
    Params:
    -block: candidate scan block.
    -universe: Shapely Geometry to be covered by block.
    -mesh: mesh coordinate points
    -weights: Airmass Factor scores at mesh coordinate points.
    -last: last selected block.
    '''
    if(block.intersects(mesh)==False or block.intersection(last).area > last.area*.04): #20000000000.):#if block doesn't have mesh points
        return float(-inf)                                                        #or if block intersects last block
    else:
        af_intx = block.intersection(mesh)
        af_intx = transform(project2, af_intx)
        intxpts = array(af_intx).round().astype(int)
        if(len(intxpts.shape)!=2):                    #shapely issue 
            return float(-inf)
        else:
            pts_arr = [50-intxpts[:,1], intxpts[:,0]+130]
            weight = (block.intersection(universe).area**2 - block.intersection(covered).area**2
                    - block.centroid.distance(last.centroid)**4)
            if(weight >=0):
                weight /= weights[pts_arr].sum()
            else:
                #weight = 0.  #zero weight, pos definite
                weight *= weights[pts_arr].sum()
            return weight

##Greedy Algorithm for Set Cover
#@input blockset : geopandas dataframe
@jit
def greedy_gc_mesh(blockset, universe_set, mesh, weights):
    '''
    This Greedy Algorithm selects scan blocks by the weight function, which utilizes the mesh inputs.
    block.
    
    Params:
    - blockset: A set of candidate scan blocks in GeoPandas.GeoDataFrame format.
    - universe_set: A Shapely Geometry object for the area required to be covered.
    - mesh: A Shapely MultiPoint object that contains the mesh coordinates to be passed to the weight function.
    - weights: The Airmass Factor scores used in the weight function calculated at 5-min intervals during the day.
    '''
    
    if(universe_set.difference(blockset.unary_union).area != 0.0):
        print('Error: Blocks do not cover universe.')
        return
    if(all(blockset.is_valid)==False):
        print('Error: invalid geometries in blocks.')
        return
    mesh_unv = mesh
    scan_blocks = blockset
    universe = universe_set
    cover_set = gpd.GeoDataFrame()
    cover_set.crs = geo_proj4
    covered = sgeom.Point()
    lastgeom = sgeom.Point()
    block_weight = []
    max_ind = []
    ii=0 #flow control index, prevents infinite looping
    t=0 #time from start in hours
    print('Greedy Algorithm commenced' )
    while( (universe.area > 490000000.)  and (ii<len(blockset)) ):
        if(universe.area > 500000000000.):   #if there is more area than a normal scan block size
            #partially apply args to calc_block_weight function
            weight_mesh_f = partial(calc_block_weight,
                                universe = universe,
                                mesh = mesh_unv,
                                weights = weights[t, :, :],
                                covered = covered,
                                last=lastgeom)
            block_weight = list(map( weight_mesh_f, scan_blocks['geometry']))
            max_ind = argmax(block_weight)
        else:
            block_weight = scan_blocks.intersection(universe).area
            max_ind = block_weight.idxmax()
        lastgeom = scan_blocks.geometry[max_ind]
        cover_set = cover_set.append(scan_blocks.iloc[max_ind])
        covered = cover_set.unary_union
        #print(ii,explain_validity(universe_set),explain_validity(covered)) #checks validity of geoms for debugging
        universe = universe.difference(covered).buffer(0)
        mesh_unv = mesh_unv.intersection(universe)
        scan_blocks = scan_blocks.drop(max_ind).reset_index(drop=True)
        print (ii+1, end=' ')
        ii+=1
        t+=1
    print('Finished.',len(cover_set),'scanning blocks chosen.')
    print('Universe Area = '+str(universe.area))
    cover_set = cover_set.reset_index(drop=True)
    return cover_set


@jit
def greedy_gc_unif(blockset, universe_set):
    '''
    This Greedy Algorithm gives uniform weight to all areas of land. Most basic weight function.
    
    Params:
    - blockset: A set of candidate scan blocks in GeoPandas.GeoDataFrame format.
    - universe_set: A Shapely Geometry object for the area required to be covered.
    '''
    if(universe_set.difference(blockset.unary_union).area != 0.0):
        print('Error: Blocks do not cover universe.')
        return
    if(all(blockset.is_valid)==False):
        print('Error: invalid geometries in blocks.')
        return
    scan_blocks = blockset
    universe = universe_set
    cover_set = gpd.GeoDataFrame()
    cover_set.crs = geo_proj4
    covered = sgeom.Point()
    lastgeom = sgeom.Point()
    block_weight = []
    max_ind = []
    ii=0 #flow control index, prevents infinite looping
    #t=0 #time from start in hours
    print('Greedy Algorithm commenced')
    while( (universe.area > 490000000.)  and (ii<len(blockset)) ):
        #if(ii%12 ==0):
        #    t+=1
        #partially apply args to calc_block_weight function
        block_weight = scan_blocks.intersection(universe).area - scan_blocks.centroid.distance(lastgeom.centroid)
        max_ind = block_weight.idxmax()
        lastgeom = scan_blocks.geometry[max_ind]
        cover_set = cover_set.append(scan_blocks.iloc[max_ind])
        covered = cover_set.unary_union
        #print(ii,explain_validity(universe_set),explain_validity(covered)) #checks validity of geoms for debugging
        universe = universe.difference(covered).buffer(0)
        scan_blocks = scan_blocks.drop(max_ind).reset_index(drop=True)
        print (ii, end=' ')
        ii+=1
    print('Finished.',len(cover_set),'scanning blocks chosen.')
    print('Universe Area = '+str(universe.area))
    cover_set = cover_set.reset_index(drop=True)
    return cover_set