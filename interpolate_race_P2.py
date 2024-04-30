#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:30:29 2024

@author: khanh
"""

# this script interpolate race in P2 table
# there only 2 year avalaible (2010 and 2020 based on census)
# interpolate to get yearly data

import pandas as pd
import numpy as np

# import P2 2010
df = pd.read_csv('/home/khanh/Documents/iSUPER_paper/historical_ej/data/race/DECENNIALPL2010.P2-Data.csv')
df = df.drop(index=0)
geoidblock2010 = np.array(df['GEO_ID'])
tract2010_temp = np.array(df['NAME'])
totalPop2010_temp = np.array(df['P002001'])
white2010_temp = np.array(df['P002005'])
tract2010 = []
totalPop2010 = []
white2010 = []
pctMinority2010 = []
for i in range(0, len(tract2010_temp)):
    temp = tract2010_temp[i].split(',')[1].split(' ')[3]
    tract2010.append(int(np.floor(float(temp))))
    totalPop2010.append(float(totalPop2010_temp[i]))
    white2010.append(float(white2010_temp[i]))
    if totalPop2010[i] > 0:
        pctMinority2010.append((totalPop2010[i]-white2010[i])/totalPop2010[i])
    else:
        pctMinority2010.append(0)

# import P2 2020
df = pd.read_csv('/home/khanh/Documents/iSUPER_paper/historical_ej/data/race/DECENNIALPL2020.P2-Data.csv')
df = df.drop(index=0)
geoidblock2020 = np.array(df['GEO_ID'])
tract2020_temp = np.array(df['NAME'])
totalPop2020_temp = np.array(df['P2_001N'])
white2020_temp = np.array(df['P2_005N'])
tract2020 = []
totalPop2020 = []
white2020 = []
pctMinority2020 = []
for i in range(0, len(tract2020_temp)):
    temp = tract2020_temp[i].split(',')[1].split(' ')[3]
    tract2020.append(int(np.floor(float(temp))))
    totalPop2020.append(float(totalPop2020_temp[i]))
    white2020.append(float(white2020_temp[i]))
    if totalPop2020[i] > 0:
        pctMinority2020.append((totalPop2020[i]-white2020[i])/totalPop2020[i])
    else:
        pctMinority2020.append(0)
    
    
count = 0
notcount = []
dMinority = []
dTotalPop = []
dWhite = []
test = []
for i in range(0, len(geoidblock2020)):
    ind = np.where(geoidblock2020[i] == geoidblock2010)
    if len(ind[0] != 0):
        # if 2010 and 2020 have the same geoid, perform linear interpolation
        ind = ind[0][0]
        dMinority.append((pctMinority2020[i] - pctMinority2010[ind])/10)
        dTotalPop.append((totalPop2020[i] - totalPop2010[ind])/10)
        dWhite.append((white2020[i] - white2010[ind])/10)
        count += 1
    elif len((np.where(tract2020[i] == np.array(tract2010)))[0]) != 0:
        # find the closest census tracts and take the average for all
        # assign to the census block
        notcount.append(tract2020[i])
        ind = np.where(tract2020[i] == np.array(tract2010))
        dMinority.append(np.average(((np.array(pctMinority2020)[i])) - np.average((np.array(pctMinority2010)[ind])))/10)
        dTotalPop.append(np.average(((np.array(totalPop2020)[i])) - np.average((np.array(totalPop2010)[ind])))/10)
        dWhite.append(np.average(((np.array(white2020)[i])) - np.average((np.array(white2010)[ind])))/10)
        # print(dMinority[count])
        count += 1
    else:
        dMinority.append(pctMinority2020[i]/10)
        dTotalPop.append(totalPop2020[i]/10)
        dWhite.append(white2020[i]/10)
        count += 1

# interpolation    
# write data for 11 years from 2020 to 2010
startYear = 2020
for i in range(0, 11):
    pctMinority = np.subtract(pctMinority2020, np.multiply((i), dMinority))
    totalPop = np.subtract(totalPop2020, np.multiply((i), dTotalPop))
    white = np.subtract(white2020, np.multiply((i), dWhite))
    for j in range(0, len(pctMinority)):
        if pctMinority[j] >= 1:
            pctMinority[j] = 1

    df = {'GEO_ID': geoidblock2020, 'P2_001N': totalPop, 'P2_005N': white, 'pctMinority': pctMinority}
    df = pd.DataFrame(df)
    df.to_csv('pctMinority_' + str(startYear - i) + '.csv', index=False)

# write data for 2 years from 2021 to 2022
startYear = 2020
for i in range(1, 3):
    pctMinority = np.add(pctMinority2020, np.multiply((i), dMinority))
    totalPop = np.add(totalPop2020, np.multiply((i), dTotalPop))
    white = np.subtract(white2020, np.multiply((i), dWhite))
    for j in range(0, len(pctMinority)):
        if pctMinority[j] >= 1:
            pctMinority[j] = 1

    df = {'GEO_ID': geoidblock2020, 'P2_001N': totalPop, 'P2_005N': white, 'pctMinority': pctMinority}
    df = pd.DataFrame(df)
    df.to_csv('pctMinority_' + str(startYear + i) + '.csv', index=False)












































