#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 09:19:28 2023

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
score = []
# for y in range(5,6):
for y in range(0, len(years)):
    # import csv MA EJ
    df = pd.read_csv('/media/khanh/SSP_Data_Vol1/iSUPER_paper/historical_ej/EJMasterData_' + str(years[y]) + '.csv')
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
    maIncome = 84385
    
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
        
    EJ = [[], [], [], []]
    for i in range(0, len(geoid)): 
        # income
        if np.isnan(pctHouseholdIncome_blockGroup[i]) != True and round(pctHouseholdIncome_blockGroup[i]) <= 65:
            EJ[0].append(1)
        else:
            EJ[0].append(0)
        # minority
        if np.isnan(pctMinority[i]) != True and round(pctMinority[i]) >= 40: # check for bflag
            EJ[1].append(1)
        else:
            EJ[1].append(0)
        # english isolation
        if np.isnan(pctLimEng[i]) != True and round(pctLimEng[i]) >= 25: # check for cflag
            EJ[2].append(1)
        else:
            EJ[2].append(0)
        # minority
        if (np.isnan(pctMinority[i]) != True and round(pctMinority[i]) >= 25 and round(pctMinority[i]) < 40) and (round(pctHouseholdIncome_municipal[i]) <= 150):
            EJ[3].append(1)
        else:
            EJ[3].append(0)
            
    EJ[1] = np.add(EJ[1], EJ[3])
    EJ = np.array(EJ[0:3])
    totalEJ = np.sum(EJ[0:3], axis=0)
    color = []
    for i in range(0, len(totalEJ)):
        if totalEJ[i] == 1:
            color.append('#1f77b4')
        elif totalEJ[i] == 2:
            color.append('yellow')
        elif totalEJ[i] == 3:
            color.append('red')
    path = '/media/khanh/SSP_Data_Vol1/iSUPER_paper/shapefiles_plotformat/ma_shapefile.shp'
    df1 = gpd.read_file(path)
    za = gpd.read_file(path)
    df1 = df1.to_crs("EPSG:4326")
                
    path = "/media/khanh/SSP_Data_Vol1/iSUPER_paper/shapefiles_plotformat/CENSUS2020BLOCKGROUPS_POLY.shp"
    
    df = gpd.read_file(path)
    df = df.to_crs("EPSG:4326")
    temp = df.GEOID20
    temp_pop = df.POP20
    temp_land = df.ALAND20/1000**2 # in m2 => convert to km2
    geoidShapefile = []
    for i in range(0, len(temp)):
        geoidShapefile.append(int(temp[i]))
    
    sortEJ = []
    ej_pop = []
    check_ej_pop = 0
    ej_land = 0
    ej1 = 0
    ej1_pop = 0
    ej1_land = 0
    ej2 = 0
    ej2_land = 0
    ej2_pop = 0
    ej3 = 0
    ej3_land = 0
    ej3_pop = 0
    cityNames = []

    for i in range(0, len(geoidShapefile)):
        ind = np.argwhere(np.array(geoid) == geoidShapefile[i])
        if len(ind) != 0:
            ind = ind[0][0]
            temp = np.hstack((EJ[:,ind], 0))
            sortEJ.append(temp)
            cityNames.append(municipalityNames[ind])
            if sum(EJ[:,ind]) >= 1:
                check_ej_pop += temp_pop[i]
                ej_land += temp_land[i]
            ej_pop.append(temp_pop[ind])
            if sum(EJ[:,ind]) == 1:
                ej1 += 1
                ej1_pop += temp_pop[i]
                ej1_land += temp_land[i]
            elif sum(EJ[:,ind]) == 2:
                ej2 += 1
                ej2_pop += temp_pop[i]
                ej2_land += temp_land[i]
            elif sum(EJ[:,ind]) == 3:
                ej3 += 1
                ej3_pop += temp_pop[i]
                ej3_land += temp_land[i]
        else:
            sortEJ.append(np.array([0, 0, 0, 1]))
            ej_pop.append(0)
            cityNames.append('nan')
            
    totalEJ = np.sum(sortEJ, axis=1)
    total_ej_pop = 0
    for i in range(0, len(totalEJ)):
        if totalEJ[i] > 0:
            total_ej_pop += ej_pop[i]
            
    c = np.array(['000', '001', '010', '100', '111', '110', '101', '011'])
    cl = ['white', 'yellow', 'green', '#1f77b4', 'purple', 'orange', 'red', 'blue']
    legend = ['None EJ', 'English Isolation', 'Minority', 'Income', 'Income, English Isolation, Minority',
              'Income, Minority', 'Income, English Isolation', 'Minority, English Isolation', 'Data Not Available']
    color = []
    
    for i in range(0, len(sortEJ)):
        if sortEJ[i][3] == 0:
            tempStr = str(sortEJ[i][0]) + str(sortEJ[i][1]) + str(sortEJ[i][2])
            color.append(cl[np.argwhere(c == tempStr)[0][0]])
        else:
            color.append('grey')
            
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
    
    legend_elements = [Line2D([0], [0], color=cl[1], label=legend[1], markerfacecolor='g', markersize=2),
                       Line2D([0], [0], color=cl[2], label=legend[2], markerfacecolor='g', markersize=2),
                       Line2D([0], [0], color=cl[3], label=legend[3], markerfacecolor='g', markersize=2),
                       Line2D([0], [0], color=cl[4], label=legend[4], markerfacecolor='g', markersize=2),
                       Line2D([0], [0], color=cl[5], label=legend[5], markerfacecolor='g', markersize=2),
                       Line2D([0], [0], color=cl[6], label=legend[6], markerfacecolor='g', markersize=2),
                       Line2D([0], [0], color=cl[7], label=legend[7], markerfacecolor='g', markersize=2),
                       Line2D([0], [0], color='grey', label=legend[8], markerfacecolor='g', markersize=2),]
    
        
    # Create the figure
    # ax.legend(handles=legend_elements, loc='lower left', fontsize = 5)
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=False, ncol=3, fontsize = 6)
    plt.title(str(years[y]) + ' Massachusetts EJ Communities based on MA Criteria', fontsize=10)
    ax.xaxis.set_major_formatter('{x:1.1f}$^\circ$')
    ax.yaxis.set_major_formatter('{x:1.1f}$^\circ$')
    plt.savefig('MA_EJ_Map' + str(years[y]) + '.png', dpi=500)
    score.append(sortEJ)
    
df = {'GEOID': geoidShapefile, 'Municipality': cityNames, '2016': score[0], '2017': score[1], '2018': score[2],
      '2019': score[3], '2020': score[4], '2021': score[5], '2022': score[6]}
df = pd.DataFrame(df)
df.to_csv('MA_EJScore.csv', index=False)