#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:13:51 2023

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

years = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
MHHI = [75297, 77385, 79835, 85843, 84385, 89645, 94488]
# years = [2016]
# MHHI = [75297]
score = []
# for y in range(0,1):
for y in range(0, len(years)):
    # import csv MA EJ
    df = pd.read_csv('/home/khanh/Documents/iSUPER_paper/historical_ej/EJMasterData_' + str(years[y]) + '.csv')
    geoid = df['GEOID20']
    pctMinority = df['PercentMinority']
    pctLimEng = df['PercentLimitedEnglishSpeaking']
    householdIncome_blockGroup = df['MedianHouseholdIncome']
    pctHouseholdIncome_blockGroup = []
    householdIncome_municipal = df['MunicipalityMedianHouseholdIncome']
    municipalityNames = df['MunicipalityNames']
    pctHouseholdIncome_municipal = []
    pop = df['POP20']
    # convert to number
    
    tempGeoid = []
    tempPctMinority = []
    tempPctLimEng = []
    tempHouseholdIncome_blockGroup = []
    tempPctHouseholdIncome_blockGroup = []
    tempHouseholdIncom_municipal = []
    tempPctHouseholdIncome_municipal = []
    maIncome = MHHI[y]
    
    for i in range(0, len(pctMinority)):
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
        if householdIncome_municipal[i] == '-':
            householdIncome_municipal[i] = math.nan
                
    for i in range(0, len(pctMinority)):
        if pctMinority[i] != '#DIV/0!' and pctLimEng[i] != '#DIV/0!' and householdIncome_blockGroup[i] != '-' and str(householdIncome_municipal[i]) != 'nan':
            if householdIncome_blockGroup[i] == '250,000+':
                householdIncome_blockGroup[i] = '250000'
            if householdIncome_municipal[i] == '250,000+':
                householdIncome_municipal[i] = '250000'
            tempPctMinority.append(np.double(pctMinority[i])*100)
            tempPctLimEng.append(np.double(pctLimEng[i])*100)
            tempHouseholdIncome_blockGroup.append(np.double(householdIncome_blockGroup[i]))
            tempPctHouseholdIncome_blockGroup.append(np.double(householdIncome_blockGroup[i])/maIncome*100)
            tempHouseholdIncom_municipal.append(np.double(householdIncome_municipal[i]))
            tempPctHouseholdIncome_municipal.append(np.double(householdIncome_municipal[i])/maIncome*100)
            tempGeoid.append(geoid[i])
            
    geoid_org = geoid
    geoid = tempGeoid
    pctMinority = tempPctMinority
    pctLimEng = tempPctLimEng
    pctHouseholdIncome_blockGroup = tempPctHouseholdIncome_blockGroup
    pctHouseholdIncome_municipal = tempPctHouseholdIncome_municipal
    
    # compute score for EJ based on a multi-criterion framework
    # DOI: 10.1007/s10668-003-4713-0
    
    minorityScore = []
    limEngScore = []
    incomeScore = []
    for i in range(0, len(pctMinority)):
        minorityScore.append((pctMinority[i] - min(pctMinority))/(max(pctMinority) - min(pctMinority))*100)
        limEngScore.append((pctLimEng[i] - min(pctLimEng))/(max(pctLimEng) - min(pctLimEng))*100)
        incomeScore.append((pctHouseholdIncome_blockGroup[i] - min(pctHouseholdIncome_blockGroup))/(max(pctHouseholdIncome_blockGroup) - min(pctHouseholdIncome_blockGroup))*100)
    
    incomeScoreOrg = incomeScore
    incomeScore = np.subtract(100, incomeScore)
    ejScore = np.add(minorityScore, limEngScore)
    ejScore = np.add(ejScore, incomeScore)
    
    # compute the score for the baseline
    minorityBase = (25 - min(pctMinority))/(max(pctMinority) - min(pctMinority))*100
    limEngBase = (25 - min(pctMinority))/(max(pctMinority) - min(pctMinority))*100
    incomeBase = 100 - (65 - min(pctMinority))/(max(pctMinority) - min(pctMinority))*100
    municipalIncomeBase = (65 - min(pctHouseholdIncome_municipal))/(max(pctHouseholdIncome_municipal) - min(pctHouseholdIncome_municipal))*100
    
    baselineEJ = minorityBase + limEngBase + incomeBase + municipalIncomeBase
    EJ = ejScore >= baselineEJ
    
    path = '/home/khanh/Documents/iSUPER_paper/shapefiles_plotformat/ma_shapefile.shp'
    df1 = gpd.read_file(path)
    za = gpd.read_file(path)
    df1 = df1.to_crs("EPSG:4326")
                
    path = "/home/khanh/Documents/iSUPER_paper/shapefiles_plotformat/CENSUS2020BLOCKGROUPS_POLY.shp"
    
    df = gpd.read_file(path)
    df = df.to_crs("EPSG:4326")
    temp = df.GEOID20
    geoidShapefile = []
    for i in range(0, len(temp)):
        geoidShapefile.append(int(temp[i]))
    
    sortEJ = []
    totalEJ = []
    new_ejScore = []
    cityNames = []
    for i in range(0, len(geoidShapefile)):
        ind = np.argwhere(np.array(geoid) == geoidShapefile[i])
        if len(ind) != 0:
            ind = ind[0][0]
            sortEJ.append(EJ[ind])
            new_ejScore.append(ejScore[ind])
            cityNames.append(municipalityNames[ind])
            if pctHouseholdIncome_blockGroup[ind] <= 65:
                totalEJ.append(1)
            else:
                totalEJ.append(0)
        else:
            sortEJ.append(0)
            new_ejScore.append('nan')
            totalEJ.append(0)
            cityNames.append('nan')
            
    cl = ['red', '#1f77b4', 'grey']
    legend = ['EJ Community', 'EJ Community with Income Criterion', 'Data Not Available']
    color = []
    for i in range(0, len(sortEJ)):
        if sortEJ[i] == True:
            color.append('red')
        else:
            if str(new_ejScore[i]) == 'nan':
                color.append('grey')
            else:
                color.append('white')
    
    # blue color for income limitation which is less than 65%
    for i in range(0, len(totalEJ)):
        if totalEJ[i] == 1:
            color[i] = '#1f77b4'
            
    # Information for each state 
    df.iloc[0,:]
    # shapefile_geoid = df['GEOID20']
    
    f,ax = plt.subplots(1,1, figsize=(8,6), sharex=True, sharey=True, dpi=500)
    # plot bold city boundaries
    df1.plot(ax=ax, edgecolor='k', linewidth=0.5, facecolor = 'none')
    
    # plot heatmap
    # find the good colorbar range
    vmax = 1
    vmin = 0
    # plt.title(names[n] + ' Emis from Point Source')
    divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%",pad=0,alpha=1)
    df.plot(ax=ax, color=color, edgecolor='k', linewidth=0.1, alpha = 0.7)
    
    divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%",pad=0,alpha=1)
    df.plot(ax=ax, color=color, edgecolor='k', linewidth=0.1, alpha = 0.7)
    
    legend_elements = [Line2D([0], [0], color=cl[0], label=legend[0], markerfacecolor='g', markersize=2),
                       Line2D([0], [0], color=cl[1], label=legend[1], markerfacecolor='g', markersize=2),
                       Line2D([0], [0], color=cl[2], label=legend[2], markerfacecolor='g', markersize=2)]
    
        
    # Create the figure
    # ax.legend(handles=legend_elements, loc='lower left', fontsize = 5)
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=False, ncol=3, fontsize = 7)
    plt.title(str(years[y]) + ' Massachusetts EJ Communities based on EJ Score System', fontsize=10)
    plt.savefig('MA_EJ_Score_Map' + str(years[y]) + '.png', dpi=500)
    score.append(new_ejScore)
    
df = {'GEOID': geoidShapefile, 'Municipality': cityNames, '2016': score[0], '2017': score[1], '2018': score[2],
      '2019': score[3], '2020': score[4], '2021': score[5], '2022': score[6]}
df = pd.DataFrame(df)
df.to_csv('EJScore.csv', index=False)
    