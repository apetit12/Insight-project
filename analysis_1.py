#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:48:11 2019

@author: antoinepetit
"""

import pandas as pd
import matplotlib.pyplot as plt
# pd.options.mode.chained_assignment = None

################################ FUNCTIONS ####################################
def has_ad(metadata):
    if 'hasAd' not in metadata:
        return 0
    else:
        idx = metadata.find('hasAd')
        if metadata[idx+7]=='f':  # no ad
            return 0
        elif metadata[idx+7]=='t':  # ad
            return 1
        else:
            raise ValueError("Should be true of false", metadata[idx+7])

def has_multiplier(metadata):
    if 'pointsMultiplier' not in metadata:
        return 0
    else:
        idx = metadata.find('pointsMultiplier')
        try:
            return int(metadata[idx+18:idx+20])
        except:
            try:
                return int(metadata[idx+18:idx+19])
            except:
                raise ValueError('wrong multiplier number', metadata[idx+16:idx+17])

################################ DATA #########################################
df_whales = pd.read_csv('bq-results-20190920-001914-whales.csv')
df_sheeps = pd.read_csv('bq-results-20190920-003615-sheeps.csv')
df_sloths = pd.read_csv('bq-results-20190920-003125-sloths.csv')

temp1 = df_whales.describe()
print(temp1)
temp2 = df_sheeps.describe()
print(temp2)
temp3 = df_sloths.describe()
print(temp3)

################################# PREPROCESSING ###############################
df_whales.loc[:,'category']=df_whales['category'].str.strip()
df_whales['multipliers'] = df_whales['metadata'].apply(lambda x: has_multiplier(x))
df_whales['ad'] = df_whales['metadata'].apply(lambda x: has_ad(x))

df_sheeps.loc[:,'category']=df_sheeps['category'].str.strip()
df_sheeps['multipliers'] = df_sheeps['metadata'].apply(lambda x: has_multiplier(x))
df_sheeps['ad'] = df_sheeps['metadata'].apply(lambda x: has_ad(x))

df_sloths.loc[:,'category']=df_sloths['category'].str.strip()
df_sloths['multipliers'] = df_sloths['metadata'].apply(lambda x: has_multiplier(x))
df_sloths['ad'] = df_sloths['metadata'].apply(lambda x: has_ad(x))


df_sheeps.groupby(['category']).count()['lifeId']
df_sloths.groupby(['category']).count()['lifeId']
df_whales.groupby(['category']).count()['lifeId']

fig1 = plt.figure(1)
ax11 = fig1.add_subplot(311)
df_whales['prizeCents'].plot.hist(ax=ax11, density=True)
plt.title('Whales')
plt.ylim([0,2.2e-7])
ax12 = fig1.add_subplot(312)
df_sheeps['prizeCents'].plot.hist(ax=ax12, density=True)
plt.title('Occasional buyers')
plt.ylim([0,2.2e-7])
ax13 = fig1.add_subplot(313)
df_sloths['prizeCents'].plot.hist(ax=ax13, density=True)
plt.title('No-buyers')
plt.xlabel('Prize value (1/100 USD)')
plt.ylim([0,2.2e-7])
plt.tight_layout()

fig2 = plt.figure(2)
ax21 = fig2.add_subplot(311)
df_whales['multipliers'].plot.hist(ax=ax21, density=True)
plt.title('Whales')
#plt.ylim([0,0.2])
#plt.xlim([0,15])
ax22 = fig2.add_subplot(312)
df_sheeps['multipliers'].plot.hist(ax=ax22, density=True)
plt.title('Occasional buyers')
#plt.ylim([0,0.2])
#plt.xlim([0,15])
ax23 = fig2.add_subplot(313)
df_sloths['multipliers'].plot.hist(ax=ax23, density=True)
plt.title('No-buyers')
plt.xlabel('')
#plt.ylim([0,0.2])
#plt.xlim([0,15])
plt.tight_layout()
