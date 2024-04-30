#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:23:36 2024

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

df = pd.read_csv('/home/khanh/Documents/iSUPER_paper/historical_ej/MA_EJScore.csv')
y16 = df['2016']
y17 = df['2017']
y18 = df['2018']
y19 = df['2019']
y20 = df['2020']
y21 = df['2021']
y22 = df['2022']

y16_income = [] # income EJ return 1, return 0 otherwise
y16_minority = [] # minority EJ return 1, return 0 otherwise
y16_english = [] # english isolation EJ return 1, return 0 otherwise
y16_data = []   # available data return 0, not available data return 1
y17_income = []
y17_minority = []
y17_english = []
y17_data = []
y18_income = []
y18_minority = []
y18_english = []
y18_data = []
y19_income = []
y19_minority = []
y19_english = []
y19_data = []
y20_income = []
y20_minority = []
y20_english = []
y20_data = []
y21_income = []
y21_minority = []
y21_english = []
y21_data = []
y22_income = []
y22_minority = []
y22_english = []
y22_data = []

income_ind = 1
minority_ind = 3
english_ind = 5
data_ind = 7

for i in range(0, len(y16)):
    if int(y16[i][data_ind]) != 1:
        y16_income.append(int(y16[i][income_ind]))
        y16_minority.append(int(y16[i][minority_ind]))
        y16_english.append(int(y16[i][english_ind]))
        y16_data.append(int(y16[i][data_ind]))
        
    if int(y17[i][data_ind]) != 1:
        y17_income.append(int(y17[i][income_ind]))
        y17_minority.append(int(y17[i][minority_ind]))
        y17_english.append(int(y17[i][english_ind]))
        y17_data.append(int(y17[i][data_ind]))
    
    if int(y18[i][data_ind]) != 1:
        y18_income.append(int(y18[i][income_ind]))
        y18_minority.append(int(y18[i][minority_ind]))
        y18_english.append(int(y18[i][english_ind]))
        y18_data.append(int(y18[i][data_ind]))
    
    if int(y19[i][data_ind]) != 1:
        y19_income.append(int(y19[i][income_ind]))
        y19_minority.append(int(y19[i][minority_ind]))
        y19_english.append(int(y19[i][english_ind]))
        y19_data.append(int(y19[i][data_ind]))
    
    if int(y20[i][data_ind]) != 1:
        y20_income.append(int(y20[i][income_ind]))
        y20_minority.append(int(y20[i][minority_ind]))
        y20_english.append(int(y20[i][english_ind]))
        y20_data.append(int(y20[i][data_ind]))
    
    if int(y21[i][data_ind]) != 1:
        y21_income.append(int(y21[i][income_ind]))
        y21_minority.append(int(y21[i][minority_ind]))
        y21_english.append(int(y21[i][english_ind]))
        y21_data.append(int(y21[i][data_ind]))
    
    if int(y22[i][data_ind]) != 1:
        y22_income.append(int(y22[i][income_ind]))
        y22_minority.append(int(y22[i][minority_ind]))
        y22_english.append(int(y22[i][english_ind]))
        y22_data.append(int(y22[i][data_ind]))

y16_ej = np.add(np.add(y16_income, y16_minority), y16_english)
y17_ej = np.add(np.add(y17_income, y17_minority), y17_english)
y18_ej = np.add(np.add(y18_income, y18_minority), y18_english)
y19_ej = np.add(np.add(y19_income, y19_minority), y19_english)
y20_ej = np.add(np.add(y20_income, y20_minority), y20_english)
y21_ej = np.add(np.add(y21_income, y21_minority), y21_english)
y22_ej = np.add(np.add(y22_income, y22_minority), y22_english)

# year - missing blocks - income EJ - minority EJ - english EJ - EJ with 2 criteria - EJ with 3 criteria - EJ communities
with open('MA_EJ_Summary.csv', 'w') as f:
    f.write('year' + ',' + 'missing_blocks' + ',' + 'income_EJ' + ',' + 'minority_EJ' + ',' + 'english_EJ'
            + ',' + '2_EJ_criteria' + ',' + '3_EJ_criteria' + ',' + 'total_EJ_communities' + ',' + '\n')
    f.write('2016' + ',' + str(len(y16) - len(y16_ej)) + ',' + str(sum(y16_income)) + ',' + str(sum(y16_minority)) + ',' + str(sum(y16_english))
            + ',' + str(sum(y16_ej == 2)) + ',' + str(sum(y16_ej == 3)) + ',' + str(sum(y16_ej != 0)) + '\n')
    f.write('2017' + ',' + str(len(y17) - len(y17_ej)) + ',' + str(sum(y17_income)) + ',' + str(sum(y17_minority)) + ',' + str(sum(y17_english))
            + ',' + str(sum(y17_ej == 2)) + ',' + str(sum(y17_ej == 3)) + ',' + str(sum(y17_ej != 0)) + '\n')
    f.write('2018' + ',' + str(len(y18) - len(y18_ej)) + ',' + str(sum(y18_income)) + ',' + str(sum(y18_minority)) + ',' + str(sum(y18_english))
            + ',' + str(sum(y18_ej == 2)) + ',' + str(sum(y18_ej == 3)) + ',' + str(sum(y18_ej != 0)) + '\n')
    f.write('2019' + ',' + str(len(y19) - len(y19_ej)) + ',' + str(sum(y19_income)) + ',' + str(sum(y19_minority)) + ',' + str(sum(y19_english))
            + ',' + str(sum(y19_ej == 2)) + ',' + str(sum(y19_ej == 3)) + ',' + str(sum(y19_ej != 0)) + '\n')
    f.write('2020' + ',' + str(len(y20) - len(y20_ej)) + ',' + str(sum(y20_income)) + ',' + str(sum(y20_minority)) + ',' + str(sum(y20_english))
            + ',' + str(sum(y20_ej == 2)) + ',' + str(sum(y20_ej == 3)) + ',' + str(sum(y20_ej != 0)) + '\n')
    f.write('2021' + ',' + str(len(y21) - len(y21_ej)) + ',' + str(sum(y21_income)) + ',' + str(sum(y21_minority)) + ',' + str(sum(y21_english))
            + ',' + str(sum(y21_ej == 2)) + ',' + str(sum(y21_ej == 3)) + ',' + str(sum(y21_ej != 0)) + '\n')
    f.write('2022' + ',' + str(len(y22) - len(y22_ej)) + ',' + str(sum(y22_income)) + ',' + str(sum(y22_minority)) + ',' + str(sum(y22_english))
            + ',' + str(sum(y22_ej == 2)) + ',' + str(sum(y22_ej == 3)) + ',' + str(sum(y22_ej != 0)) + '\n')