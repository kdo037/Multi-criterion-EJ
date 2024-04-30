#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:05:43 2024

@author: khanh
"""

import csv
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pykrige.ok import OrdinaryKriging
import matplotlib as mcolors
from math import radians, cos, sin, asin, sqrt

# this script interpolate the design value ozone in Greater Boston

lines = []
with open('/media/khanh/SSP_Data_Vol1/iSUPER_paper/GreaterBoston_O3_designValue.csv', 'r', encoding = 'utf-8', errors = 'ignore') as readFile:
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

fig, ax = plt.subplots()
df=gpd.read_file("/media/khanh/SSP_Data_Vol1/iSUPER_paper/shapefiles_plotformat/ma_shapefile.shp")
df.plot(ax = ax, color="white", edgecolor='black')
norm = mcolors.colors.TwoSlopeNorm(vmin=np.min(z), vcenter=(np.max(z)+np.min(z))/2, vmax=np.max(z))
mesh = plt.pcolormesh(lon_interp,lat_interp,z, norm=norm, cmap="jet", alpha=0.8)
cbar = fig.colorbar(mesh)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# plt.clim(10, 12)
cbar.mappable.set_clim(np.min(z),np.max(z))
plt.xlim(xlim[0], xlim[1])
plt.ylim(ylim[0], ylim[1])
plt.scatter(-np.abs(lon), lat)
plt.title('O$_3$ Design Values - Kriging Interpolation')
cbar.set_label('O$_3$ Concentrations [ppb]')
# plt.savefig('ozone_interp.png', dpi=500)
# # 
pc = plt.scatter(-np.abs(lon), lat, c=o3, s=60, cmap='jet')
plt.clim(np.min(z),np.max(z))
ax.xaxis.set_major_formatter('{x:1.1f}$^\circ$')
ax.yaxis.set_major_formatter('{x:1.1f}$^\circ$')
plt.savefig('ozone_interp.png', dpi=500)

mesh_lon, mesh_lat = np.meshgrid(lon_interp, np.sort(lat_interp))

# step 2: import census combine
df = pd.read_csv('/media/khanh/SSP_Data_Vol1/iSUPER_paper/census_combine.csv')
c_white = df['Total_White']
c_total = df['Total_Population']
c_lat = df['LAT']
c_lon = df['LON']
c_tract = df['TractID']
c_geoid = df['GEOID']
c_total_poverty_survey = df['Total_Poverty_Survey']
c_50PovertyLevel = df['50PovertyLevel']
c_poverty_ratio = df['Poverty_Ratio']
c_median_house_income = df['Median_Household_Income']
c_total_welfare_survey = df['totalWelfareSurvey']
c_foodstamp = df['foodstamp']
c_no_assistance = df['No_Assistance']

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
pm25_f = []
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
    # dlon = np.subtract(rmesh_lon1d, rmesh_lon1d[10])
    # dlat = np.subtract(rmesh_lat1d, rmesh_lat1d[10])
    
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

    dis_tol = 0.07 # 0.07 km = 70 m
    # assign ozone design value value to census tract if found distance less than 1km
    if min(dis) < 0.07 and c_total[i] > 0:
        white_f.append(c_white[i])
        total_f.append(c_total[i])
        tract_f.append(c_tract[i])
        lat_f.append(c_lat[i])
        lon_f.append(c_lon[i])
        geoid_f.append(c_geoid[i])
        pm25_f.append(mesh_z1d[min_ind[0]])
        total_poverty_survey_f.append(c_total_poverty_survey[i])
        poverty_level50_f.append(c_50PovertyLevel[i])
        poverty_ratio_f.append(c_poverty_ratio[i])
        median_house_income_f.append(c_median_house_income[i])
        total_welfare_survey_f.append(c_total_welfare_survey[i])
        foodstamp_f.append(c_foodstamp[i])
        no_assistance_f.append(c_no_assistance[i])
        
########## plot cumulative share for 50 poverty level #######################
# sort advantage from least to most
advantage = np.subtract(1,poverty_ratio_f) # percent of not poverty
ratio_sort_ind = np.argsort(advantage)
# advatage ascending = disadvantage descending
advantage_asc = advantage[ratio_sort_ind]
pm25_asc = np.array(pm25_f)[ratio_sort_ind]

#############################################################
temp = pm25_asc
# pm25_asc = np.multiply(pm25_asc, 0)
# pm25_asc = np.add(pm25_asc, 7)
temp_total_f = np.array(total_f)[ratio_sort_ind]
pm25_asc = np.multiply(temp-min(temp), temp_total_f)
#############################################################

plt.scatter(advantage_asc, pm25_asc)

cum_advantage_pm25 = []
cum_advantage = []
for i in range(0, len(pm25_asc)):
    cum_advantage_pm25.append(sum(pm25_asc[0:i])/sum(pm25_asc))
    cum_advantage.append(sum(temp_total_f[0:i])/sum(temp_total_f))
    # cum_white_pm25.append(sum(pm25_f[0:i])/sum(pm25_f))
    # cum_white.append(sum(white_f[0:i])/sum(white_f))
    
fig = plt.figure()
ax = fig.add_subplot()
# plt.plot(cum_nonwhite, cum_pm25, '--')
plt.plot(cum_advantage, cum_advantage_pm25, '--r')
plt.plot([0, 1], [0, 1], 'k-')
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend(['O$_3$ Inequality Curve', 'Equality Line'])
plt.ylabel('Cumulative share of O$_3$')
plt.xlabel('Cumulative share of population \n (most disadvantaged from left)')
plt.title('O$_3$ Inequality by 50% Poverty Level')
ax.set_aspect('equal', adjustable='box')
plt.savefig('inequality_curve_50_poverty_level.jpg',bbox_inches='tight', dpi=500)

########## plot cumulative share for 50 poverty level #######################

advantage_asc = advantage_asc
pm25_asc = temp
nbins = 10
count = 1
bins = []
freq = []
conc = []
temp = []
temp_conc = []
# x_temp = np.linspace(1, nbins, nbins)/nbins
x_temp = np.linspace(1, nbins, nbins)/nbins
x = []
for i in range(0, len(x_temp)):
    x.append(str(x_temp[i]))


for i in range(0, len(advantage_asc)):
    if i <= len(advantage_asc)*count/nbins:
        temp.append(advantage_asc[i])
        temp_conc.append(pm25_asc[i])
    else:
        bins.append(str(int(np.mean(temp)*100)))
        freq.append(len(temp))
        conc.append(np.mean(temp_conc))
        count += 1
        temp = []
        temp_conc = []
        temp.append(advantage_asc[i])
        temp_conc.append(pm25_asc[i])
bins.append(str(int(np.mean(temp)*100)))
freq.append(len(temp))
conc.append(np.mean(temp_conc))

plt.figure()
plt.bar(bins, conc)
plt.ylim([min(conc) - 2*(max(conc) - min(conc)), max(conc) + (max(conc) - min(conc))])
plt.xlabel('50% Poverty Level')
plt.ylabel('O$_3$ Design Value [ppb]')
plt.title('By 50% Poverty Level')
plt.xticks(rotation = 30)
bbox_inches='tight'
plt.savefig('bar_50_poverty_level.jpg',bbox_inches='tight', dpi=500)

########## plot cumulative share for 50 poverty level #######################
# compute the non-white ratio
non_white_f = np.subtract(total_f, white_f)
non_white_ratio_f = non_white_f/total_f
ratio_sort_ind = np.argsort(non_white_ratio_f)
ratio_sort_ind_desc = ratio_sort_ind[::-1]

non_white_ratio_desc = non_white_ratio_f[ratio_sort_ind_desc]
non_white_ratio_asc = non_white_ratio_f[ratio_sort_ind]
# non_white_ratio_desc = non_white_ratio_desc[0:250]

pm25_desc = np.array(pm25_f)[ratio_sort_ind_desc]
pm25_asc = np.array(pm25_f)[ratio_sort_ind]
# pm25_desc = pm25_desc[0:250]

# average every 10% of the data
percent = 0.1
cummulative_pm25 = []
cummulative_non_white = []
for i in range(0, 10):
    cummulative_pm25.append(np.mean(pm25_desc[i*20:(i+1)*20]))
    cummulative_non_white.append(np.mean(non_white_ratio_desc[i*20:(i+1)*20]))

plt.plot(cummulative_non_white, cummulative_pm25)

# sort by non-white population
pop_sort_ind = np.argsort(non_white_f)
pop_sort_ind_desc = pop_sort_ind[::-1]
non_white_desc = non_white_f[pop_sort_ind_desc]
pm25_pop_desc = np.array(pm25_f)[pop_sort_ind_desc]
# sort by white population
# pop_sort_w_ind = np.argsort(white_f)
# pop_sort_w_ind_desc = pop_sort_w_ind[::-1]
# white_acs = np.array(white_f)[pop_sort_w_ind]
# pm25_pop_acs = np.array(pm25_f)[pop_sort_w_ind]

############################################################################
pop_sort_w_ind = np.argsort(1-non_white_ratio_f)
white_acs = np.array(white_f)[pop_sort_w_ind]
total_acs = np.array(total_f)[pop_sort_w_ind]
pm25_pop_acs = np.array(pm25_f)[pop_sort_w_ind]
temp = pm25_pop_acs
pm25_pop_acs = pm25_pop_acs - min(pm25_pop_acs)
pm25_pop_acs = np.multiply(pm25_pop_acs, total_acs)
############################################################################

cum_white_pm25 = []
cum_white = []
white_ratio_desc = 1 - non_white_ratio_desc
white_ratio_asc = 1 - non_white_ratio_asc

for i in range(0, len(pm25_desc)):
    cum_white_pm25.append(sum(pm25_pop_acs[0:i])/sum(pm25_pop_acs))
    cum_white.append(sum(total_acs[0:i])/sum(total_acs))
    # cum_white_pm25.append(sum(pm25_f[0:i])/sum(pm25_f))
    # cum_white.append(sum(white_f[0:i])/sum(white_f))
    
fig = plt.figure()
ax = fig.add_subplot()
# plt.plot(cum_nonwhite, cum_pm25, '--')
plt.plot(cum_white, cum_white_pm25, '--r')
plt.plot([0, 1], [0, 1], 'k-')
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend(['O$_3$ Inequality Curve', 'Equality Line'])
plt.ylabel('Cumulative share of differences in O$_3$')
plt.xlabel('Cumulative share of population \n (most disadvantaged from left)')
plt.title('O$_3$ Inequality by White Population')
ax.set_aspect('equal', adjustable='box')
plt.savefig('inequality_curve_white_population.jpg',bbox_inches='tight', dpi=500)
########## plot cumulative share for 50 poverty level #######################
advantage_asc = 1-non_white_ratio_desc
pm25_asc = temp
nbins = 10
count = 1
bins = []
freq = []
conc = []
temp = []
temp_conc = []
# x_temp = np.linspace(1, nbins, nbins)/nbins
x_temp = np.linspace(1, nbins, nbins)/nbins
x = []
for i in range(0, len(x_temp)):
    x.append(str(x_temp[i]))


for i in range(0, len(advantage_asc)):
    if i <= len(advantage_asc)*count/nbins:
        temp.append(advantage_asc[i])
        temp_conc.append(pm25_asc[i])
    else:
        # print(temp)
        bins.append(str(int(np.mean(temp)*100)))
        freq.append(len(temp))
        conc.append(np.mean(temp_conc))
        count += 1
        temp = []
        temp_conc = []
        temp.append(advantage_asc[i])
        temp_conc.append(pm25_asc[i])
bins.append(str(int(np.mean(temp)*100)))
freq.append(len(temp))
conc.append(np.mean(temp_conc))

plt.figure()
plt.bar(bins, conc)
plt.ylim([min(conc) - 2*(max(conc) - min(conc)), max(conc) + (max(conc) - min(conc))])
plt.xlabel('Percentage of White Population')
plt.ylabel('O$_3$ Design Value [ppb]')
plt.title('By Percentage White Population') # in each census tract
plt.xticks(rotation = 30)
bbox_inches='tight'
plt.savefig('bar_white_population.jpg',bbox_inches='tight', dpi=500)

# sort advantage from least to most
advantage = np.array(median_house_income_f) # percent of not poverty
ratio_sort_ind = np.argsort(advantage)
# advatage ascending = disadvantage descending
advantage_asc = advantage[ratio_sort_ind]
temp_total_f = np.array(total_f)[ratio_sort_ind]
pm25_asc = np.array(pm25_f)[ratio_sort_ind]

#############################################################
temp = pm25_asc
# pm25_asc = np.multiply(pm25_asc, 0)
# pm25_asc = np.add(pm25_asc, 7)
pm25_asc = np.multiply(temp-min(temp), temp_total_f)
#############################################################

plt.scatter(advantage_asc, pm25_asc)

cum_advantage_pm25 = []
cum_advantage = []
for i in range(0, len(pm25_asc)):
    cum_advantage_pm25.append(sum(pm25_asc[0:i])/sum(pm25_asc))
    cum_advantage.append(sum(temp_total_f[0:i])/sum(temp_total_f))
    # cum_white_pm25.append(sum(pm25_f[0:i])/sum(pm25_f))
    # cum_white.append(sum(white_f[0:i])/sum(white_f))
    
fig = plt.figure()
ax = fig.add_subplot()
# plt.plot(cum_nonwhite, cum_pm25, '--')
plt.plot(cum_advantage, cum_advantage_pm25, '--r')
plt.plot([0, 1], [0, 1], 'k-')
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend(['O$_3$ Inequality Curve', 'Equality Line'])
plt.ylabel('Cumulative share of O$_3$')
plt.title('O$_3$ Inequality by Median Household Income')
plt.xlabel('Cumulative share of population \n (most disadvantaged from left)')
ax.set_aspect('equal', adjustable='box')
plt.savefig('inequality_curve_median_income.jpg',bbox_inches='tight', dpi=500)

########## plot cumulative share for median household salary ################

# create x = bins, y = number of occurences, z = concentrations
pm25_asc = temp
nbins = 10
count = 1
bins = []
freq = []
conc = []
temp = []
temp_conc = []
# x_temp = np.linspace(1, nbins, nbins)/nbins
x_temp = np.linspace(1, nbins, nbins)/nbins
x = []
for i in range(0, len(x_temp)):
    x.append(str(x_temp[i]))


for i in range(0, len(advantage_asc)):
    if i <= len(advantage_asc)*count/nbins:
        temp.append(advantage_asc[i])
        temp_conc.append(pm25_asc[i])
    else:
        # print(temp)
        bins.append(str(int(np.mean(temp))))
        freq.append(len(temp))
        conc.append(np.mean(temp_conc))
        count += 1
        temp = []
        temp_conc = []
        temp.append(advantage_asc[i])
        temp_conc.append(pm25_asc[i])
bins.append(str(int(np.mean(temp))))
freq.append(len(temp))
conc.append(np.mean(temp_conc))

plt.figure()
plt.bar(bins, conc)
plt.ylim([min(conc) - 2*(max(conc) - min(conc)), max(conc) + (max(conc) - min(conc))])
plt.xlabel('Median Household Income [USD]')
plt.ylabel('O$_3$ Design Value [ppb]')
plt.title('By Median Household Income')
plt.xticks(rotation = 30)
bbox_inches='tight'
plt.savefig('bar_median_income.jpg',bbox_inches='tight', dpi=500)

########## plot cumulative share for Foodstamp and Public Assistance ########
no_assistance_ratio = np.divide(no_assistance_f, total_welfare_survey_f)
advantage = np.array(no_assistance_ratio) # percent of not poverty
ratio_sort_ind = np.argsort(advantage)
# advatage ascending = disadvantage descending
advantage_asc = advantage[ratio_sort_ind]
temp_total_f = np.array(total_f)[ratio_sort_ind]
pm25_asc = np.array(pm25_f)[ratio_sort_ind]

#############################################################
temp = pm25_asc
# pm25_asc = np.multiply(pm25_asc, 0)
# pm25_asc = np.add(pm25_asc, 7)
pm25_asc = np.multiply(temp-min(temp), temp_total_f)
#############################################################

plt.figure()
plt.scatter(advantage_asc, pm25_asc)

cum_advantage_pm25 = []
cum_advantage = []
for i in range(0, len(pm25_asc)):
    cum_advantage_pm25.append(sum(pm25_asc[0:i])/sum(pm25_asc))
    cum_advantage.append(sum(temp_total_f[0:i])/sum(temp_total_f))
    # cum_white_pm25.append(sum(pm25_f[0:i])/sum(pm25_f))
    # cum_white.append(sum(white_f[0:i])/sum(white_f))
    
fig = plt.figure()
ax = fig.add_subplot()
# plt.plot(cum_nonwhite, cum_pm25, '--')
plt.plot(cum_advantage, cum_advantage_pm25, '--r')
plt.plot([0, 1], [0, 1], 'k-')
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend(['O$_3$', 'Equality Line'])
plt.ylabel('Cumulative share of O$_3$')
plt.xlabel('Cumulative share of population \n (most disadvantaged from left)')
plt.title('O$_3$ Inequality by SNAP')
ax.set_aspect('equal', adjustable='box')
plt.savefig('inequality_curve_snap.jpg',bbox_inches='tight', dpi=500)
########## plot cumulative share for Foodstamp and Public Assistance ########
# create x = bins, y = number of occurences, z = concentrations
pm25_asc = temp
nbins = 10
count = 1
bins = []
freq = []
conc = []
temp = []
temp_conc = []
x_temp = np.linspace(1, nbins, nbins)/nbins
x = []
for i in range(0, len(x_temp)):
    x.append(str(x_temp[i]))


for i in range(0, len(advantage_asc)):
    if i <= len(advantage_asc)*count/nbins:
        temp.append(advantage_asc[i])
        temp_conc.append(pm25_asc[i])
    else:
        # print(temp)
        bins.append(str(int(np.mean(temp)*100)))
        freq.append(len(temp))
        conc.append(np.mean(temp_conc))
        count += 1
        temp = []
        temp_conc = []
        temp.append(advantage_asc[i])
        temp_conc.append(pm25_asc[i])
bins.append(str(int(np.mean(temp)*100)))
freq.append(len(temp))
conc.append(np.mean(temp_conc))

plt.figure()
plt.bar(bins, conc)
plt.ylim([min(conc) - 2*(max(conc) - min(conc)), max(conc) + (max(conc) - min(conc))])
plt.xlabel('Percentage of No SNAP')
plt.ylabel('O$_3$ Design Value [ppb]')
plt.title('By Percentage of SNAP')
plt.xticks(rotation = 30)
bbox_inches='tight'
plt.savefig('bar_snap.jpg',bbox_inches='tight', dpi=500)
