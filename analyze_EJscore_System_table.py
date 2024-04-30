#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 21:43:53 2024

@author: khanh
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import csv

city_interest = ["Boston city", "Chelsea city", "Brookline town", "Cambridge city", "Newton city", "Worcester city",
                  "Lynn city", "Saugus town", "Everett city"]

df = pd.read_csv('/home/khanh/Documents/iSUPER_paper/historical_ej/EJScore.csv')
t16 = df['2016']
t17 = df['2017']
t18 = df['2018']
t19 = df['2019']
t20 = df['2020']
t21 = df['2021']
t22 = df['2022']
tgeoid = df['GEOID']
tcityName = df['Municipality']

y16 = []
y17 = []
y18 = []
y19 = []
y20 = []
y21 = []
y22 = []
geoid = []
cityName = []
# remove nan for all the years
for i in range(0, len(t16)):
    if (str(t16[i]) != 'nan' and str(t17[i]) != 'nan' and  str(t18[i]) != 'nan' and
        str(t17[i]) != 'nan' and str(t18[i]) != 'nan' and str(t19[i]) != 'nan' and
        str(t20[i]) != 'nan' and str(t21[i]) != 'nan' and str(t22[i]) != 'nan'):
            y16.append(t16[i])
            y17.append(t17[i])
            y18.append(t18[i])
            y19.append(t19[i])
            y20.append(t20[i])
            y21.append(t21[i])
            y22.append(t22[i])
            geoid.append(tgeoid[i])
            cityName.append(tcityName[i])
            
city_interest = np.unique(cityName)
    
avg16 = []
avg17 = []
avg18 = []
avg19 = []
avg20 = []
avg21 = []
avg22 = []
for i in range(0, len(city_interest)):
    t16 = []
    t17 = []
    t18 = []
    t19 = []
    t20 = []
    t21 = []
    t22 = []
    for j in range(0, len(cityName)):
        if cityName[j] == city_interest[i]:
            t16.append(y16[j])
            t17.append(y17[j])
            t18.append(y18[j])
            t19.append(y19[j])
            t20.append(y20[j])
            t21.append(y21[j])
            t22.append(y22[j])
    avg16.append(np.average(t16))
    avg17.append(np.average(t17))
    avg18.append(np.average(t18))
    avg19.append(np.average(t19))
    avg20.append(np.average(t20))
    avg21.append(np.average(t21))
    avg22.append(np.average(t22))

years = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
avg = np.vstack((avg16, avg17, avg18, avg18, avg20, avg21, avg22))

with open ('EJ_Score_All_Municipality.csv', 'w') as f:
    f.write('Municipality' + ',' + '2016' + ',' + '2017' + ',' + '2018'
            + ',' + '2019' + ',' + '2020' + ',' + '2021' + ',' + '2022' + '\n')
    for i in range(0, len(avg[0])):
        f.write(city_interest[i] + ',' + str(avg[0][i]) + ',' + str(avg[1][i]) + ',' + str(avg[2][i])
                + ',' + str(avg[3][i]) + ',' + str(avg[4][i]) + ',' + str(avg[5][i]) + ',' + str(avg[6][i]) + '\n')


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(1,1,1)
for i in range(0, len(city_interest)):
    plt.plot(years, avg[:,i], '-o')

plt.axhline(y = 91.43, color = 'k', linestyle = '--') 

city_interest = np.append(city_interest, 'Baseline')

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                  box.width, box.height * 0.9])
ax.legend(city_interest, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=14, fontsize = 14)
plt.ylabel('Environmental Justice Score', fontsize = 17)
plt.title('Environmental Justice Trends in Greater Boston', fontsize = 20)
plt.savefig('EJ_scores.png', dpi=700)