# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import pandas as pd;
import numpy as np;
import torch as pt;  #pytorch

import matplotlib.pyplot as plt

# read data into df and sort out the data 

#infile=r'H:\Documents\ArcGIS\RS\CELL_COUNTS.csv'

infile=r'H:\Documents\ArcGIS\RS\VIIRS_matchups_0720.csv'


df=pd.read_csv (infile,header=1)    # skipe the first row 

df=pd.read_csv (infile) 
#extract data based on c0lumn number or headers



#LAT	LON	YEAR	JDAY	TIME	Rrs_410-1_day	Rrs_443-1_day	Rrs_486-1_day	Rrs_551-1_day	DATE	TIME	ZONE	DEPTH	LAT	LON	KB COUNTS	PROOFED?

headers=df.head(0)

lats=np.asarray(df.iloc[:,0]).astype(float)
lons=np.asarray(df.iloc[:,1]) .astype(float)
years=np.asarray(df.iloc[:,2]) .astype(float)
ydays=np.asarray(df.iloc[:,3]) .astype(float)


#previous day data before the in situ samples
#skip column 4
Rrs410_pd=np.asarray(df.iloc[:,5]).astype(float)
Rrs443_pd=np.asarray(df.iloc[:,6]).astype(float)
Rrs486_pd=np.asarray(df.iloc[:,7]).astype(float)
Rrs551_pd=np.asarray(df.iloc[:,8]).astype(float)
nlw638_pd=np.asarray(df.iloc[:,9]).astype(float)
Rrs671_pd=np.asarray(df.iloc[:,10]).astype(float)
Chla_pd=np.asarray(df.iloc[:,11]).astype(float)

#skip column 12
Rrs410_sd=np.asarray(df.iloc[:,13]).astype(float)
Rrs443_sd=np.asarray(df.iloc[:,14]).astype(float)
Rrs486_sd=np.asarray(df.iloc[:,15]).astype(float)
Rrs551_sd=np.asarray(df.iloc[:,16]).astype(float)
nlw638_sd=np.asarray(df.iloc[:,17]).astype(float)
Rrs671_sd=np.asarray(df.iloc[:,18]).astype(float)
Chla_sd=np.asarray(df.iloc[:,19]).astype(float)


#skip column 2o
Rrs410_ad=np.asarray(df.iloc[:,21]).astype(float)
Rrs443_ad=np.asarray(df.iloc[:,22]).astype(float)
Rrs486_ad=np.asarray(df.iloc[:,23]).astype(float)
Rrs551_ad=np.asarray(df.iloc[:,24]).astype(float)
nlw638_ad=np.asarray(df.iloc[:,25]).astype(float)
Rrs671_ad=np.asarray(df.iloc[:,26]).astype(float)
Chla_ad=np.asarray(df.iloc[:,27]).astype(float)


#skip column 28
dates=df.iloc[:,29]
tiems=df.iloc[:,30]
zones=df.iloc[:,31]
depth=np.asarray(df.iloc[:,32]).astype(float)
insitu_lats=np.asarray(df.iloc[:,33]).astype(float)
insitu_lons=np.asarray(df.iloc[:,34]).astype(float)
cell_counts=np.asarray(df.iloc[:,35]).astype(float)
proofed=df.iloc[:,36]


# source out 


idx=np.where((Rrs410_sd>0) & (Rrs486_sd>0) & (Rrs551_sd>0) & (nlw638_sd>0) & (Rrs671_sd>0) & (cell_counts>0))
idx=idx[0]

good_Rrs410_2=Rrs410_sd[idx]
good_Rrs486_2=Rrs486_sd[idx]
good_Rrs551_2=Rrs551_sd[idx]
good_nlw638_2=nlw638_sd[idx]
good_Rrs671_2=Rrs671_sd[idx]
good_Chla_2=Chla_sd[idx]

# counts
good_lats_2=insitu_lats[idx]
good_lons_2=insitu_lons[idx]
good_cell_counts_2=cell_counts[idx]

#output the sorted data out to check the location


dd2=np.column_stack((good_Rrs410_2,good_Rrs486_2,good_Rrs551_2,good_nlw638_2,good_Rrs671_2,good_lats_2,good_lons_2,good_cell_counts_2))

# which modle to use
#



dd2=pd.DataFrame(dd2)
dd2.to_csv('sorted_same_day_data_v2.csv')
