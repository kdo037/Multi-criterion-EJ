#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:01:27 2023

@author: khanh
"""

import csv
import numpy as np
import shapefile 
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pykrige.ok import OrdinaryKriging
import matplotlib as mcolors
from math import radians, cos, sin, asin, sqrt
import math

# this script interpolate the design value ozone in Greater Boston

lines = []
with open('/home/khanh/Documents/iSUPER_paper/GreaterBoston_O3_designValue.csv', 'r', encoding = 'utf-8', errors = 'ignore') as readFile:
    reader = csv.reader(readFile)
    lines = list(reader)

lat = []
lon = []
o3 = []
for i in range(1, len(lines)):
    lat.append(float(lines[i][1]))
    lon.append(float(lines[i][2]))
    o3.append(float(lines[i][0]))
    
lat_interp = np.linspace(min(lat), max(lat), 250)
lon_interp = np.linspace(min(-np.absolute(lon)), max(-np.absolute(lon)), 250)

# Kriging interpolation
OK = OrdinaryKriging(
    (-np.absolute(lon)),
    (lat),
    (o3),
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
)

z, ss = OK.execute("grid", lon_interp, lat_interp)

# find the closest point based on cencus tract location and the heatmap to 
# assign ozone design value conc to the cencus tract

# step 1: create a mesh lat and mesh lon
mesh_lon, mesh_lat = np.meshgrid(lon_interp, np.sort(lat_interp))

fig, ax = plt.subplots()
norm = mcolors.colors.TwoSlopeNorm(vmin=np.min(z), vcenter=(np.max(z)+np.min(z))/2, vmax=np.max(z))
mesh = ax.pcolormesh(lon_interp,lat_interp,z, norm=norm, cmap="jet", alpha=0.8)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
plt.close()

df=gpd.read_file("/home/khanh/Documents/iSUPER_paper/shapefiles_plotformat/ma_shapefile.shp")
df.plot(color="white", edgecolor='black')
norm = mcolors.colors.TwoSlopeNorm(vmin=np.min(z), vcenter=(np.max(z)+np.min(z))/2, vmax=np.max(z))
mesh = plt.pcolormesh(lon_interp,lat_interp,z, norm=norm, cmap="jet", alpha=0.8)
cbar = fig.colorbar(mesh)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# plt.clim(10, 12)
cbar.mappable.set_clim(np.min(z),np.max(z))
plt.xlim(xlim[0], xlim[1])
plt.ylim(ylim[0], ylim[1])
plt.scatter(-np.abs(lon), lat)
plt.title('O$_3$ Design Values - Kriging Interpolation')
cbar.set_label('O$_3$ Concentrations [\u03BCg m$^{-3}$]')
# plt.savefig('ozone_interp.png', dpi=500)

pc = plt.scatter(-np.abs(lon), lat, c=o3, s=60, cmap='jet')
plt.clim(np.min(z),np.max(z))
# plt.savefig('ozone_interp.png', dpi=500)

mesh_lon, mesh_lat = np.meshgrid(lon_interp, np.sort(lat_interp))


# import csv MA EJ
df = pd.read_csv('/home/khanh/Documents/iSUPER_paper/EJMasterData.csv')
c_lat = df['INTPTLAT20']
c_lon = df['INTPTLON20']
geoid = df['GEOID20']
pctMinority = df['PercentMinority']
pctLimEng = df['PercentLimitedEnglishSpeaking']
householdIncome_blockGroup = df['MedianHouseholdIncome']
pctHouseholdIncome_blockGroup = []
householdIncome_municipal = df['MunicipalityMedianHouseholdIncome']
pctHouseholdIncome_municipal = []
pop = df['POP20']

# step 4: search for closest coordinates from census track to the heat map
mesh_lon1d = np.reshape(mesh_lon, (len(mesh_lon)*len(mesh_lon))) # 2d to 1d array
rmesh_lon1d = [radians(s) for s in mesh_lon1d] # convert to radians

mesh_lat1d = np.reshape(mesh_lat, (len(mesh_lat)*len(mesh_lat))) # 2d to 1d array
rmesh_lat1d = [radians(s) for s in mesh_lat1d] # convert to radians

mesh_z1d = np.reshape(z, (len(z)*len(z)))

white_f = []
total_f = []
tract_f = []
lat_f = []
lon_f = []
geoid_f = []
o3_f = []
total_poverty_survey_f = []
poverty_level50_f = []
poverty_ratio_f = []
median_house_income_f = []
total_welfare_survey_f = []
foodstamp_f = []
no_assistance_f = []
for i in range(0, len(c_lon)):
    dlon = np.subtract(rmesh_lon1d, radians(c_lon[i]))
    dlat = np.subtract(rmesh_lat1d, radians(c_lat[i]))

    term1 = [sin(s) for s in (dlat/2)]
    term1 = np.power(term1, 2)
    term2 = cos(radians(c_lat[i]))
    term3 = [cos(s) for s in rmesh_lat1d]
    term4 = [sin(s) for s in (dlon/2)]
    term4 = np.power(term4, 2)
    a = term1 + np.multiply(term2, term3)*term4
    c = 2*np.arcsin(np.sqrt(a))
    r = 6371
    dis = c*r
    
    # find the minimum index
    min_ind = np.argsort(dis)

    dis_tol = 0.3 # 0.07 km = 70 m
    # assign PM2.5 value to census tract if found distance less than 70m
    if min(dis) < dis_tol:
        lat_f.append(c_lat[i])
        lon_f.append(c_lon[i])
        o3_f.append(mesh_z1d[min_ind[0]])
    else:
        lat_f.append(c_lat[i])
        lon_f.append(c_lon[i])
        o3_f.append(math.nan)
    
    
df['O3'] = o3_f

df = pd.DataFrame(df)
df.to_csv('EJMasterDataWithAQO3.csv', index = False)
    
o3 = []
for i in range(0, len(o3_f)):
    if np.isnan(o3_f[i]) == False:
        o3.append(o3_f[i])