#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:44:48 2024

@author: khanh
"""


import csv
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import matplotlib as mcolors
from math import radians, cos, sin, asin, sqrt

low_memory = False

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# NOTE THAT 278053 is an outlier
locations = ['362907', '365983', '590664', '686842', '483527', '660765', '755485',
              '67141', '70247',
              '69326', '324885', '69319', '65681', '274165', '236037',
              '231932', '224410', '66919', '71721',
              '71723', '65009',
              '228878', '341151', '69577', '228934', '65107',
              '69254', '354604', '228624', '71001']

additionalLocations = ['928534', '323403', '356676', '219455',
                        '362990', '362937', '363449', '394267', '362939', '363452', '362941',
                        '227365', '354269', '354416', 'c1', 'c2']

# additionalLocations = ['1700956', '928534', '332935', '323403', '356676', '219455', '1183714', '221475', '342334',
#                        '362908', '362990', '362937', '363449', '394267', '362939', '1542178', '363452',
#                        '362941', '1418211', '227365', '1068840', '1150086', '354269', '1723152', '278053',
#                        '332888', '354416']

# additionalLocations = ['928534', '323403', '356676', '219455', '221475', '342334',
#                        '362990', '362937', '363449', '394267', '362939', '363452', '362941',
#                        '227365', '354269', '278053', '332888', '354416', 'c1', 'c2', 'MOD-PM-00530_MODULAIR-PM',
#                        'MOD-PM-00531_MODULAIR-PM', 'MOD-PM-00549_MODULAIR-PM', 'SN000-049_ARISense-v200', 'SN000-072_ARISense-v200',
#                        'MOD-PM-00552_MODULAIR-PM', 'SN000-062_ARISense-v200', 'MOD-PM-00141_MODULAIR-PM', 'MOD-00028_MODULAIR']
# # 928534 frrom May 2023
# # 323403 from 2022
# # 356676 from Sep 2022
# # 219455 from 2021
# # 221475 from 2021
# # 342334 from May 2022
# # 362908 from Sep 2022
# # 362990 from Sep 2022
# # 362937 from Sep 2022
# # 363449 from Sep 2022
# # 394267 from Jan 2023
# # 362939 from Jan 2023
# # 363452 from Sep 2022
# # 362941 from Sep 2022
# # 227365 from Jun 2022
# # 354269 from July 2022
# # 278053 from Jan 2022
# # 332888 from May 2022
# # 354416 from Jul 2022

locations = locations + additionalLocations

# locations = ['354416', 'c1', 'c2', 'MOD-PM-00530_MODULAIR-PM',
#                        'MOD-PM-00531_MODULAIR-PM', 'MOD-PM-00549_MODULAIR-PM', 'SN000-049_ARISense-v200', 'SN000-072_ARISense-v200',
#                        'MOD-PM-00552_MODULAIR-PM', 'SN000-062_ARISense-v200', 'MOD-PM-00141_MODULAIR-PM', 'MOD-00028_MODULAIR']
# locations = ['MOD-PM-00530_MODULAIR-PM',
#                        'MOD-PM-00531_MODULAIR-PM', 'SN000-049_ARISense-v200', 'SN000-072_ARISense-v200',
#                        'MOD-PM-00552_MODULAIR-PM', 'SN000-062_ARISense-v200', 'MOD-PM-00141_MODULAIR-PM', 'MOD-00028_MODULAIR']
# lookup and record the lat and lon
lat = []
lon = []
for loc in range(0, len(locations)):
    lines = []
    with open('/media/khanh/SSP_Data_Vol1/iSUPER_paper/ML_ready/' + locations[loc] + '.csv', 'r', encoding = 'utf-8', errors = 'ignore') as readFile:
        reader = csv.reader(readFile)
        lines = list(reader)
    lat.append(float(lines[100][7][0:7]))
    lon.append(float(lines[100][8][0:8]))
        
common_d = []
common_t = []

# Find common date and time
for loc in range(0, len(locations)):
    lines = []
    count = 0
    with open('/media/khanh/SSP_Data_Vol1/iSUPER_paper/ML_ready/' + locations[loc] + '.csv', 'r', encoding = 'utf-8', errors = 'ignore') as readFile:
        reader = csv.reader(readFile)
        lines = list(reader)
    
    d = []
    t = []  
    temp_d = []
    temp_t = []
    for i in range(1, len(lines)):
        d.append(lines[i][1])
        t.append(lines[i][2])
    if loc == 0:
        common_d = d
        common_t = t
    else:
        for i in range(0, len(common_d)):
            for j in range(0, len(d)):
                if common_d[i] == d[j] and common_t[i] == t[j]:
                    temp_d.append(common_d[i])
                    temp_t.append(common_t[i])
                    count += 1
        common_d = temp_d
        common_t = temp_t
    print(loc)
    print('loc: ' + locations[loc] + ', count: ' + str(count), ', len(common_d): ' + str(len(common_d)))
    count = 0

# Get data for 15 ML sites
d = [[] for _ in range(len(locations))]
t = [[] for _ in range(len(locations))]
o3 = [[] for _ in range(len(locations))]
obs_o3 = [[] for _ in range(len(locations))]

for loc in range(0, len(locations)):
    lines = []
    with open('/media/khanh/SSP_Data_Vol1/iSUPER_paper/ML_ready/' + locations[loc] + '.csv', 'r', encoding = 'utf-8', errors = 'ignore') as readFile:
        reader = csv.reader(readFile)
        lines = list(reader)
    temp_d = []
    temp_t = []
    temp_o3 = []
    temp_obs_o3 = []
    for i in range(1, len(lines)):
            temp_d.append(lines[i][1])
            temp_t.append(lines[i][2])
            temp_o3.append(float(lines[i][4]))
            temp_obs_o3.append(float(lines[i][4]))
    for i in range(0, len(common_d)):
        for j in range(0, len(lines)-1):
            if common_d[i] == temp_d[j] and common_t[i] == temp_t[j]:
                d[loc].append(temp_d[j])
                t[loc].append(temp_t[j])
                o3[loc].append(temp_o3[j])
                obs_o3[loc].append(temp_obs_o3[j])#!/usr/bin/env python3

# d[0][3054] to d[0][3231] from 2023-09-07 to 2023-09-16
shp = np.shape(d)
avg_o3 = []
for i in range(0, shp[0]):
    # avg_o3.append(np.mean(o3[i][3054:3231]))
    # avg_o3.append(np.mean(o3[i][2000:2177]))
    # avg_o3.append(np.mean(o3[i][1137:1314]))
    avg_o3.append(np.mean(o3[i][:]))

# # compute the maximum ozone
# site_avg_o3 = np.mean(o3, axis=0)
# ind = np.argsort((site_avg_o3))
# sort_avg_o3 = site_avg_o3[ind]
# avg_o3 = []
# for i in range(0, shp[0]):
#     temp = []
#     for j in range(len(ind)-344, len(ind)):
#         temp.append(o3[i][ind[j]])
#     avg_o3.append(np.mean(temp))
    
#reshape
new_avg_o3 = []
new_lat = []
new_lon = []
count = 0
for i in range(0, shp[0]):
    if i != 7 and i != 10:
        new_avg_o3.append(avg_o3[i])
        new_lat.append(lat[i])
        new_lon.append(lon[i])
        count += 1
avg_o3 = new_avg_o3
lat = new_lat
lon = -np.abs(new_lon)

# interp
lat_interp = np.linspace(min(lat), max(lat), 250)
lon_interp = np.linspace(min(lon), max(lon), 250)

# Kriging interpolation
OK = OrdinaryKriging(
    (lon),
    (lat),
    (avg_o3),
    variogram_model="gaussian",
    verbose=False,
    enable_plotting=False,
)

z, ss = OK.execute("grid", lon_interp, lat_interp)

# find the closest point based on cencus tract location and the heatmap to 
# assign PM2.5 conc to the cencus tract

# step 1: create a mesh lat and mesh lon
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
    # assign PM2.5 value to census tract if found distance less than 1km
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
plt.legend(['PM$_{2.5}$ Inequality Curve', 'Equality Line'])
plt.ylabel('Cumulative share of differences in PM$_{2.5}$')
plt.xlabel('Cumulative share of population \n (most disadvantaged from left)')
plt.title('PM$_{2.5}$ Inequality by 50% Poverty Level')
ax.set_aspect('equal', adjustable='box')
plt.savefig('inequality_curve_50_poverty_level.jpg',bbox_inches='tight', dpi=500)

########## plot cumulative share for 50 poverty level #######################
pm25_asc = temp
advantage_asc = advantage_asc
pm25_asc = pm25_asc
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
plt.ylabel('PM$_{2.5}$ [$\mu$g m$^{-3}$]')
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
pop_sort_w_ind = np.argsort(1-non_white_ratio_f)
# pop_sort_w_ind = np.argsort(white_f)
pop_sort_w_ind_desc = pop_sort_w_ind[::-1]
total_acs = np.array(total_f)[pop_sort_w_ind]
pm25_pop_acs = np.array(pm25_f)[pop_sort_w_ind]
temp = pm25_pop_acs

########################################################
# pm25_pop_acs = np.multiply(pm25_pop_acs, 0)
# pm25_pop_acs = np.add(pm25_pop_acs, 7)
pm25_pop_acs = pm25_pop_acs - min(pm25_pop_acs)
pm25_pop_acs = np.multiply(pm25_pop_acs, total_acs)
########################################################

# white_acs = np.array(white_f)[pop_sort_w_ind]
# pm25_pop_acs = np.array(pm25_f)[pop_sort_w_ind]
    
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
plt.legend(['PM$_{2.5}$ Inequality Curve', 'Equality Line'])
plt.ylabel('Cumulative share of differences in PM$_{2.5}$')
plt.xlabel('Cumulative share of population \n (most disadvantaged from left)')
plt.title('PM$_{2.5}$ Inequality by White Population')
ax.set_aspect('equal', adjustable='box')
plt.savefig('inequality_curve_white_population.jpg',bbox_inches='tight', dpi=500)
########## plot cumulative share for 50 poverty level #######################
advantage_asc = 1-non_white_ratio_desc
pm25_asc = temp
################################## Working on to determine the locations for least exposure group ##
# sort by increasing white population
lat_asc = np.array(lat_f)[pop_sort_w_ind] # **
lon_asc = np.array(lon_f)[pop_sort_w_ind] # **

nbins = 10
count = 1
bins = []
freq = []
conc = []
temp = []
temp_conc = []
lat_bin_w = [] # **
lon_bin_w = [] # **
temp_lat = [] # **
temp_lon = [] # **

# x_temp = np.linspace(1, nbins, nbins)/nbins
x_temp = np.linspace(1, nbins, nbins)/nbins
x = []
for i in range(0, len(x_temp)):
    x.append(str(x_temp[i]))


for i in range(0, len(advantage_asc)):
    if i <= len(advantage_asc)*count/nbins:
        temp.append(advantage_asc[i])
        temp_conc.append(pm25_asc[i])
        temp_lat.append(lat_asc[i]) # **
        temp_lon.append(lon_asc[i]) # **
    else:
        # print(temp)
        bins.append(str(int(np.mean(temp)*100)))
        freq.append(len(temp))
        conc.append(np.mean(temp_conc))
        lat_bin_w.append(temp_lat) # **
        lon_bin_w.append(temp_lon) # **
        count += 1
        temp = []
        temp_lat = [] # **
        temp_lon = [] # **
        temp_conc = []
        temp.append(advantage_asc[i])
        temp_conc.append(pm25_asc[i])
        temp_lat.append(lat_asc[i]) # **
        temp_lon.append(lon_asc[i]) # **
        
bins.append(str(int(np.mean(temp)*100)))
freq.append(len(temp))
conc.append(np.mean(temp_conc))
lat_bin_w.append(temp_lat) # **
lon_bin_w.append(temp_lon) # **
        
plt.figure()
plt.bar(bins, conc)
plt.ylim([min(conc) - 2*(max(conc) - min(conc)), max(conc) + (max(conc) - min(conc))])
plt.xlabel('Percentage of White Population')
plt.ylabel('PM$_{2.5}$ [$\mu$g m$^{-3}$]')
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
plt.legend(['PM$_{2.5}$ Inequality Curve', 'Equality Line'])
plt.ylabel('Cumulative share of differences in PM$_{2.5}$')
plt.title('PM$_{2.5}$ Inequality by Median Household Income')
plt.xlabel('Cumulative share of population \n (most disadvantaged from left)')
ax.set_aspect('equal', adjustable='box')
plt.savefig('inequality_curve_median_income.jpg',bbox_inches='tight', dpi=500)

########## plot cumulative share for median household salary ################

# create x = bins, y = number of occurences, z = concentrations
pm25_asc = temp
################################## Working on to determine the locations for least exposure group ##
# sort by increasing white population
lat_asc = np.array(lat_f)[ratio_sort_ind] # **
lon_asc = np.array(lon_f)[ratio_sort_ind] # **

nbins = 10
count = 1
bins = []
freq = []
conc = []
temp = []
temp_conc = []
lat_bin_income = [] # **
lon_bin_income = [] # **
temp_lat = [] # **
temp_lon = [] # **

# x_temp = np.linspace(1, nbins, nbins)/nbins
x_temp = np.linspace(1, nbins, nbins)/nbins
x = []
for i in range(0, len(x_temp)):
    x.append(str(x_temp[i]))


for i in range(0, len(advantage_asc)):
    if i <= len(advantage_asc)*count/nbins:
        temp.append(advantage_asc[i])
        temp_conc.append(pm25_asc[i])
        temp_lat.append(lat_asc[i]) # **
        temp_lon.append(lon_asc[i]) # **
    else:
        # print(temp)
        bins.append(str(int(np.mean(temp))))
        freq.append(len(temp))
        conc.append(np.mean(temp_conc))
        lat_bin_income.append(temp_lat) # **
        lon_bin_income.append(temp_lon) # **
        count += 1
        temp = []
        temp_lat = [] # **
        temp_lon = [] # **
        temp_conc = []
        temp.append(advantage_asc[i])
        temp_conc.append(pm25_asc[i])
        temp_lat.append(lat_asc[i]) # **
        temp_lon.append(lon_asc[i]) # **
        
bins.append(str(int(np.mean(temp))))
freq.append(len(temp))
conc.append(np.mean(temp_conc))
lat_bin_income.append(temp_lat) # **
lon_bin_income.append(temp_lon) # **

plt.figure()
plt.bar(bins, conc)
plt.ylim([min(conc) - 2*(max(conc) - min(conc)), max(conc) + (max(conc) - min(conc))])
plt.xlabel('Median Household Income [USD]')
plt.ylabel('PM$_{2.5}$ [$\mu$g m$^{-3}$]')
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
pm25_asc = np.array(pm25_f)[ratio_sort_ind]

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
plt.legend(['PM$_{2.5}$', 'Equality Line'])
plt.ylabel('Cumulative share of differences in PM$_{2.5}$')
plt.xlabel('Cumulative share of population \n (most disadvantaged from left)')
plt.title('PM$_{2.5}$ Inequality by SNAP')
ax.set_aspect('equal', adjustable='box')
plt.savefig('inequality_curve_snap.jpg',bbox_inches='tight', dpi=500)
########## plot cumulative share for Foodstamp and Public Assistance ########
# create x = bins, y = number of occurences, z = concentrations
pm25_asc = temp
################################## Working on to determine the locations for least exposure group ##
# sort by increasing white population
lat_asc = np.array(lat_f)[ratio_sort_ind] # **
lon_asc = np.array(lon_f)[ratio_sort_ind] # **

nbins = 10
count = 1
bins = []
freq = []
conc = []
temp = []
temp_conc = []
lat_bin_snap = [] # **
lon_bin_snap = [] # **
temp_lat = [] # **
temp_lon = [] # **

x_temp = np.linspace(1, nbins, nbins)/nbins
x = []
for i in range(0, len(x_temp)):
    x.append(str(x_temp[i]))


for i in range(0, len(advantage_asc)):
    if i <= len(advantage_asc)*count/nbins:
        temp.append(advantage_asc[i])
        temp_conc.append(pm25_asc[i])
        temp_lat.append(lat_asc[i]) # **
        temp_lon.append(lon_asc[i]) # **
    else:
        # print(temp)
        bins.append(str(int(np.mean(temp)*100)))
        freq.append(len(temp))
        conc.append(np.mean(temp_conc))
        lat_bin_snap.append(temp_lat) # **
        lon_bin_snap.append(temp_lon) # **
        count += 1
        temp = []
        temp_lat = [] # **
        temp_lon = [] # **
        temp_conc = []
        temp.append(advantage_asc[i])
        temp_conc.append(pm25_asc[i])
        temp_lat.append(lat_asc[i]) # **
        temp_lon.append(lon_asc[i]) # **
        
bins.append(str(int(np.mean(temp)*100)))
freq.append(len(temp))
conc.append(np.mean(temp_conc))
lat_bin_snap.append(temp_lat) # **
lon_bin_snap.append(temp_lon) # **

plt.figure()
plt.bar(bins, conc)
plt.ylim([min(conc) - 2*(max(conc) - min(conc)), max(conc) + (max(conc) - min(conc))])
plt.xlabel('Percentage of No SNAP')
plt.ylabel('PM$_{2.5}$ [$\mu$g m$^{-3}$]')
plt.title('By Percentage of SNAP')
plt.xticks(rotation = 30)
bbox_inches='tight'
plt.savefig('bar_snap.jpg',bbox_inches='tight', dpi=500)

xlim = [min(lon_interp), max(lon_interp)]
ylim = [min(lat_interp), max(lat_interp)]

fig, ax = plt.subplots()

df=gpd.read_file("/media/khanh/SSP_Data_Vol1/iSUPER_paper/shapefiles_plotformat/ma_shapefile.shp")
df.plot(ax = ax, color="white", edgecolor='black')
za = df
norm = mcolors.colors.TwoSlopeNorm(vmin=np.min(z), vcenter=(np.max(z)+np.min(z))/2, vmax=np.max(z))
mesh = plt.pcolormesh(lon_interp,lat_interp,z, norm=norm, cmap="jet", alpha=0.8)
cbar = plt.colorbar(mesh)
# plt.clim(10, 12)
cbar.mappable.set_clim(np.min(z),np.max(z))

## Ploting city names
za["center"] = za["geometry"].centroid
za_points = za.copy()
za_points.set_geometry("center", inplace = True)
xmin = xlim[0]
xmax = xlim[1]
ymin = ylim[0]
ymax = ylim[1]
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
texts = []

for x, y, label in zip(za_points.geometry.x, za_points.geometry.y, za_points["TOWN"]):
    if xmin < x and xmax > x + 0.1:
        if ymin < y and ymax > y:
            print(x, y)
            texts.append(plt.text(x, y, label, fontsize = 7))
                
plt.xlim(xlim[0], xlim[1])
plt.ylim(ylim[0], ylim[1])
# plt.scatter(lon, lat)
plt.title('Average PM$_{2.5}$ Concentrations - Kriging')
cbar.set_label('PM$_{2.5}$ Concentrations [\u03BCg m$^{-3}$]')
# plt.savefig('pm25_interp.png', dpi=500)

pc = plt.scatter(lon, lat, c=avg_o3, s=40, cmap='jet')
# pc = plt.scatter(lon_f, lat_f, c=pm25_f, s=40, cmap='jet')
plt.clim(np.min(z),np.max(z))
# plt.clim(np.min(pm25_f),np.max(pm25_f))
ax.xaxis.set_major_formatter('{x:1.1f}$^\circ$')
ax.yaxis.set_major_formatter('{x:1.1f}$^\circ$')
plt.savefig('pm25_interp.png', dpi=500)
    
    
# plot the location of highest white, lowest white #########################################
fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(wspace=0, top=0.952, right=0.9, left=.15, bottom=0.225)
ax = fig.add_subplot(1,1,1)
df=gpd.read_file("/media/khanh/SSP_Data_Vol1/iSUPER_paper/shapefiles_plotformat/ma_shapefile.shp")
df.plot(ax=ax, color="white", edgecolor='black')
## Ploting city names
za["center"] = za["geometry"].centroid
za_points = za.copy()
za_points.set_geometry("center", inplace = True)
xmin = xlim[0]
xmax = xlim[1]
ymin = ylim[0]
ymax = ylim[1]
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
texts = []

for x, y, label in zip(za_points.geometry.x, za_points.geometry.y, za_points["TOWN"]):
    if xmin < x and xmax > x + 0.1:
        if ymin < y and ymax > y:
            print(x, y)
            texts.append(plt.text(x, y, label, fontsize = 7))
                
plt.xlim(xlim[0], xlim[1])
plt.ylim(ylim[0], ylim[1])

# lowest white population
ax.scatter(lon_bin_w[0], lat_bin_w[0])
# highest white population
ax.scatter(lon_bin_w[9], lat_bin_w[9], color='darkorange')
ax.scatter(lon_bin_w[9], lat_bin_w[9], color='darkorange')
plt.title('Distribution of 10% of Lowest and 10% Highest Minority Census Tracts')
# plt.legend(['10% Highest Minority', '10% Lowest Minority'])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax.legend(['10% Highest Minority', '10% Lowest Minority', 'Income with Lowest PM$_{2.5}$'], loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3, fontsize = 9)
ax.xaxis.set_major_formatter('{x:1.1f}$^\circ$')
ax.yaxis.set_major_formatter('{x:1.1f}$^\circ$')
plt.savefig('minority_distribution.png', dpi=500)

# plot the location of highest snap, lowest snap #########################################
fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(wspace=0, top=0.952, right=0.9, left=.15, bottom=0.225)
ax = fig.add_subplot(1,1,1)
df=gpd.read_file("/media/khanh/SSP_Data_Vol1/iSUPER_paper/shapefiles_plotformat/ma_shapefile.shp")
df.plot(ax=ax, color="white", edgecolor='black')
## Ploting city names
za["center"] = za["geometry"].centroid
za_points = za.copy()
za_points.set_geometry("center", inplace = True)
xmin = xlim[0]
xmax = xlim[1]
ymin = ylim[0]
ymax = ylim[1]
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
texts = []

for x, y, label in zip(za_points.geometry.x, za_points.geometry.y, za_points["TOWN"]):
    if xmin < x and xmax > x + 0.1:
        if ymin < y and ymax > y:
            print(x, y)
            texts.append(plt.text(x, y, label, fontsize = 7))
                
plt.xlim(xlim[0], xlim[1])
plt.ylim(ylim[0], ylim[1])
# lowest snap
ax.scatter(lon_bin_snap[0], lat_bin_snap[0])
# highest snap
ax.scatter(lon_bin_snap[9], lat_bin_snap[9])
# lowest pm2.5 conc
ax.scatter(lon_bin_snap[6], lat_bin_snap[6])
# plt.legend(['10% Lowest SNAP', '10% Highest SNAP', 'SNAP with Lowest PM$_{2.5}$'])
plt.title('Distribution of 10% of Lowest and 10% Highest SNAP Census Tracts')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax.legend(['10% Highest SNAP', '10% Lowest SNAP', 'SNAP with Lowest PM$_{2.5}$'], loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3, fontsize = 9)
ax.xaxis.set_major_formatter('{x:1.1f}$^\circ$')
ax.yaxis.set_major_formatter('{x:1.1f}$^\circ$')
plt.savefig('snap_distribution.png', dpi=500)

# plot the location of highest income, lowest income #########################################
fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(wspace=0, top=0.952, right=0.9, left=.15, bottom=0.225)
ax = fig.add_subplot(1,1,1)
df=gpd.read_file("/media/khanh/SSP_Data_Vol1/iSUPER_paper/shapefiles_plotformat/ma_shapefile.shp")
df.plot(ax=ax, color="white", edgecolor='black')
## Ploting city names
za["center"] = za["geometry"].centroid
za_points = za.copy()
za_points.set_geometry("center", inplace = True)
xmin = xlim[0]
xmax = xlim[1]
ymin = ylim[0]
ymax = ylim[1]
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
texts = []

for x, y, label in zip(za_points.geometry.x, za_points.geometry.y, za_points["TOWN"]):
    if xmin < x and xmax > x + 0.1:
        if ymin < y and ymax > y:
            print(x, y)
            texts.append(plt.text(x, y, label, fontsize = 7))
                
plt.xlim(xlim[0], xlim[1])
plt.ylim(ylim[0], ylim[1])
# lowest income
ax.scatter(lon_bin_income[0], lat_bin_income[0])
# lowest pm2.5 conc
ax.scatter(lon_bin_income[5], lat_bin_income[5])
# highest income
ax.scatter(lon_bin_income[9], lat_bin_income[9])
# plt.legend(['10% Lowest Income', '10% Highest Income', 'Income with Lowest PM$_{2.5}$'])
plt.title('Distribution of 10% of Lowest and 10% Highest Income Census Tracts')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax.legend(['10% Lowest Income', '10% Highest Income', 'Income with Lowest PM$_{2.5}$'], loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3, fontsize = 9)
ax.xaxis.set_major_formatter('{x:1.1f}$^\circ$')
ax.yaxis.set_major_formatter('{x:1.1f}$^\circ$')
plt.savefig('income_distribution.png', dpi=500)
    
    
    
    
    
    
    
    
    
    