#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:38:24 2024

@author: khanh
"""

import pandas as pd
import numpy as np

# this scripts process the limited English speaking
years = [2016, 2017, 2018, 2019, 2020, 2021, 2022]

for y in range(0, len(years)):
    df = pd.read_csv('/home/khanh/Documents/iSUPER_paper/historical_ej/data/english_limited/ACSDT5Y' + str(years[y]) + '.C16002-Data.csv')
    df = df.drop(index=0)
    geoid = df['GEO_ID']
    C16002_004E = np.array(df['C16002_004E'], int)
    C16002_007E = np.array(df['C16002_007E'], int)
    C16002_010E = np.array(df['C16002_010E'], int)
    C16002_013E = np.array(df['C16002_013E'], int)
    C16002_001E = np.array(df['C16002_001E'], int)
    
    english_limited = C16002_004E + C16002_007E + C16002_010E + C16002_013E
    total = C16002_001E
    pctLimEngSpk = english_limited / total
    pctLimEngSpk = np.nan_to_num(pctLimEngSpk, nan = 0)
    
    df = {'GEO_ID': geoid, 'C16002_004E': C16002_004E, 'C16002_007E': C16002_007E, 'C16002_010E': C16002_010E,
          'C16002_013E': C16002_013E, 'english_limited': english_limited, 'C16002_001E': C16002_001E, 'pctLimEngSpk': pctLimEngSpk}
    df = pd.DataFrame(df)
    df.to_csv('limitedEngSpk_' + str(years[y]) + '.csv', index=False)