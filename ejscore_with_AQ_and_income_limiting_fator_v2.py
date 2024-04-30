#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:36:27 2023

@author: khanh

This scripts generate a new MA EJ population map with census tract resolution
New EJ map based on 2 score:
    1. thresholds as stated in MA EJ website (https://www.mass.gov/info-details/environmental-justice-populations-in-massachusetts)
        a. the annual median household income is 65 percent or less of the statewide annual median household income
            where the statewide annual median household income
        b. minorities make up 40% or more of the population
        c. 25% or more of households identify as speaking English less than very well
        d. minorities make up 25% or more of the population and the annual median household income of the municipality
          in which the neighborhood is located does not exceed 150% of the statewide annual median household income
    2. Multi-criterion EJ Map
        a. a base line with six criteria: O3 = 70 ppb, PM2.5 = 12 ug/m3, 1a, 1b, 1c, and 1d
        b. use normalization technique to get normalized impact matrix 100*[(actual value - min value)/(max value - min value)]
        c. compute normalized impact matrix accounting for minimization objectives (100 - normalized indicator score)
        d. sum all six criteria
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D

# import csv MA EJ
df = pd.read_csv('/media/khanh/SSP_Data_Vol1/iSUPER_paper/EJMasterDataWithO3PM25.csv')
geoid = df['GEOID20']
pctMinority = df['PercentMinority']
pctLimEng = df['PercentLimitedEnglishSpeaking']
householdIncome_blockGroup = df['MedianHouseholdIncome']
pctHouseholdIncome_blockGroup = []
householdIncome_municipal = df['MunicipalityMedianHouseholdIncome']
pctHouseholdIncome_municipal = []
pop = df['POP20']
o3 = df['O3']
pm25 = df['PM25']
# convert to number

tempGeoid = []
tempPctMinority = []
tempPctLimEng = []
tempHouseholdIncome_blockGroup = []
tempPctHouseholdIncome_blockGroup = []
tempHouseholdIncom_municipal = []
tempPctHouseholdIncome_municipal = []
maIncome = 84385

for i in range(0, len(pctMinority)):
    # population needs to be greater than 50
    if pop[i] < 50: 
        pctMinority[i] = math.nan
        pctLimEng[i] = math.nan
        householdIncome_blockGroup[i] = math.nan

for i in range(0, len(pctMinority)):
    if pctMinority[i] == '#DIV/0!':
        pctMinority[i] = math.nan
    if pctLimEng[i] == '#DIV/0!':
        pctLimEng[i] = math.nan
    if householdIncome_blockGroup[i] == '-':
        householdIncome_blockGroup[i] = math.nan
        
for i in range(0, len(pctMinority)):
    if pctMinority[i] != '#DIV/0!' and pctLimEng[i] != '#DIV/0!' and householdIncome_blockGroup[i] != '-':
        if householdIncome_blockGroup[i] == '250,000+':
            householdIncome_blockGroup[i] = '250000'
        tempPctMinority.append(np.double(pctMinority[i])*100)
        tempPctLimEng.append(np.double(pctLimEng[i])*100)
        tempHouseholdIncome_blockGroup.append(np.double(householdIncome_blockGroup[i]))
        tempPctHouseholdIncome_blockGroup.append(np.double(householdIncome_blockGroup[i])/maIncome*100)
        tempHouseholdIncom_municipal.append(np.double(householdIncome_municipal[i]))
        tempPctHouseholdIncome_municipal.append(np.double(householdIncome_blockGroup[i])/np.double(householdIncome_municipal[i])*100)
        tempGeoid.append(geoid[i])
        
geoid_org = geoid
geoid = tempGeoid
pctMinority = tempPctMinority
pctLimEng = tempPctLimEng
pctHouseholdIncome_blockGroup = tempPctHouseholdIncome_blockGroup
pctHouseholdIncome_municipal = tempPctHouseholdIncome_municipal

# compute score for EJ based on a multi-criterion framework
# DOI: 10.1007/s10668-003-4713-0

o3Conc = o3
pm25Conc = pm25

minorityScore = []
limEngScore = []
incomeScore = []
municipalIncomeScore = []
municipalityIncomeScore = []
o3Score = []
pm25Score = []
count = 0
# (actual - min)/(max - min)
for i in range(0, len(pctMinority)):
    # minorityScore.append((pctMinority[i] - min(pctMinority))/(max(pctMinority) - min(pctMinority))*100)
    # limEngScore.append((pctLimEng[i] - min(pctLimEng))/(max(pctLimEng) - min(pctLimEng))*100)
    # incomeScore.append(100 - (pctHouseholdIncome_blockGroup[i] - min(pctHouseholdIncome_blockGroup))/(max(pctHouseholdIncome_blockGroup) - min(pctHouseholdIncome_blockGroup))*100)
    # municipalIncomeScore.append(100 - (pctHouseholdIncome_municipal[i] - min(pctHouseholdIncome_municipal))/(max(pctHouseholdIncome_municipal) - min(pctHouseholdIncome_municipal))*100)

    # o3Score.append((o3Conc[i] - min(o3.dropna()))/(max(o3.dropna()) - min(o3.dropna()))*100)
    # pm25Score.append((pm25Conc[i] - min(pm25Conc.dropna()))/(max(pm25Conc.dropna()) - min(pm25Conc.dropna()))*100)
    o3Score.append((o3Conc[i] - 0)/(max(o3.dropna()) - 0)*100)
    pm25Score.append((pm25Conc[i] - 0)/(max(pm25Conc.dropna()) - 0)*100)
    minorityScore_temp = (pctMinority[i] - min(pctMinority))/(max(pctMinority) - min(pctMinority))*100
    limEngScore.append((pctLimEng[i] - min(pctLimEng))/(max(pctLimEng) - min(pctLimEng))*100)
    incomeScore.append((pctHouseholdIncome_blockGroup[i] - min(pctHouseholdIncome_blockGroup))/(max(pctHouseholdIncome_blockGroup) - min(pctHouseholdIncome_blockGroup))*100)
    municipalityIncomeScore.append((pctHouseholdIncome_municipal[i] - min(pctHouseholdIncome_municipal))/(max(pctHouseholdIncome_municipal) - min(pctHouseholdIncome_municipal))*100)
    if pctMinority[i] >= 25 and pctMinority[i] <= 40:
        municipalityIncomeScore20pct = (pctHouseholdIncome_municipal[i] - min(pctHouseholdIncome_municipal))/(max(pctHouseholdIncome_municipal) - min(pctHouseholdIncome_municipal))*100
        minorityScore.append((minorityScore_temp + municipalityIncomeScore20pct)/2)
        count += 1
    else:
        minorityScore.append((pctMinority[i] - min(pctMinority))/(max(pctMinority) - min(pctMinority))*100)

# o3Conc = o3.dropna()
# pm25Conc = pm25.dropna()
    
incomeScoreOrg = incomeScore
incomeScore = np.subtract(100, incomeScore)
ejScore = np.add(minorityScore, limEngScore)
ejScore = np.add(ejScore, incomeScore)
temp1 = np.add(o3Score, pm25Score)
temp2 = np.divide(temp1, 2)
ejScore = np.add(ejScore, temp2)

minorityBase = (40 - min(pctMinority))/(max(pctMinority) - min(pctMinority))*100
limEngBase = (25 - min(pctLimEng))/(max(pctLimEng) - min(pctLimEng))*100
# incomeBase = 100 - (65 - min(pctMinority))/(max(pctMinority) - min(pctMinority))*100
incomeBase = (65 - min(pctHouseholdIncome_blockGroup))/(max(pctHouseholdIncome_blockGroup) - min(pctHouseholdIncome_blockGroup))*100
# incomeBase = 65
municipalIncomeBase = (150 - min(pctHouseholdIncome_municipal))/(max(pctHouseholdIncome_municipal) - min(pctHouseholdIncome_municipal))*100
# o3Base = 70
# pm25Base = 12
# o3Base = (70 - min(o3.dropna()))/(max(o3.dropna()) - min(o3.dropna()))*100
# pm25Base = (12 - min(pm25Conc.dropna()))/(max(pm25Conc.dropna()) - min(pm25Conc.dropna()))*100

o3Base = (70 - 0)/(max(o3.dropna()) - 0)*100
pm25Base = (12 - 0)/(max(pm25Conc.dropna()) - 0)*100

baselineEJ = (minorityBase + limEngBase + (100-incomeBase) + (o3Base + pm25Base)/2)*0.65
EJ = ejScore >= baselineEJ

path = '/media/khanh/SSP_Data_Vol1/iSUPER_paper/shapefiles_plotformat/ma_shapefile.shp'
df1 = gpd.read_file(path)
za = gpd.read_file(path)
df1 = df1.to_crs("EPSG:4326")
            
path = "/media/khanh/SSP_Data_Vol1/iSUPER_paper/shapefiles_plotformat/CENSUS2020BLOCKGROUPS_POLY.shp"

df = gpd.read_file(path)
df = df.to_crs("EPSG:4326")
temp = df.GEOID20
geoidShapefile = []
for i in range(0, len(temp)):
    geoidShapefile.append(int(temp[i]))

sortEJ = []
totalEJ = []
na = []
for i in range(0, len(geoidShapefile)):
    ind = np.argwhere(np.array(geoid) == geoidShapefile[i])
    if len(ind) != 0:
        ind = ind[0][0]
        sortEJ.append(EJ[ind])
        if pctHouseholdIncome_blockGroup[ind] <= 65:
            totalEJ.append(1)
        else:
            totalEJ.append(0)
        if str(temp2[ind]) != 'nan':
            na.append(1)
        else:
            na.append(0)
    else:
        sortEJ.append(0)
        totalEJ.append(0)
        na.append(0)
        
cl = ['red', '#1f77b4', 'lightgray']
legend = ['EJ Community', 'EJ Community with Income Criterion', 'Data Not Available']
color = []
for i in range(0, len(sortEJ)):
    if sortEJ[i] == True:
        color.append('red')
    else:
        color.append('white')

# blue color for income limitation which is less than 65%
for i in range(0, len(totalEJ)):
    if totalEJ[i] == 1:
        color[i] = '#1f77b4'
    if na[i] == 0:
        color[i] = 'lightgray'
        
# Information for each state 
df.iloc[0,:]
# f,ax = plt.subplots(1,1, figsize=(8,6), sharex=True, sharey=True, dpi=300)
fig = plt.figure(figsize=(7,7))
fig.subplots_adjust(wspace=0, top=0.952, right=0.9, left=.15, bottom=0.225)
ax = fig.add_subplot(1,1,1)
# plot bold city boundaries
df1.plot(ax=ax, edgecolor='k', linewidth=0.5, facecolor = 'none')

divider = make_axes_locatable(ax)
df.plot(ax=ax, color=color, edgecolor='k', linewidth=0.1, alpha = 0.7)

divider = make_axes_locatable(ax)
df.plot(ax=ax, color=color, edgecolor='k', linewidth=0.1, alpha = 0.7)

legend_elements = [Line2D([0], [0], color=cl[0], label=legend[0], markerfacecolor='g', markersize=2),
                   Line2D([0], [0], color=cl[1], label=legend[1], markerfacecolor='g', markersize=2),
                   Line2D([0], [0], color=cl[2], label=legend[2], markerfacecolor='g', markersize=2)]


ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.07),
          fancybox=True, shadow=False, ncol=2, fontsize = 7)
plt.xlim([-71.30, -70.94])
plt.ylim([42.28, 42.60])
plt.title('Greater Boston EJ Map with Integrated PM$_{2.5}$ and O$_3$')
ax.xaxis.set_major_formatter('{x:1.1f}$^\circ$')
ax.yaxis.set_major_formatter('{x:1.1f}$^\circ$')
plt.savefig('EJ_with_AQ.png', dpi=500)

