#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this code is intended to plot maps for individual events and stations used for the source inversion
# and the focal mechanism.

import gzip
import os, sys, glob, datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from obspy.imaging.mopad_wrapper import beach
import pyasdf
from PIL import Image


# read in topo data (on a regular lat/lon grid)
# (SRTM data from: http://srtm.csi.cgiar.org/)
#with rar.open("SRTM_W_250m_ASC.rar") as fp:
################# skip the first 8 lines ###############
#    srtm = np.loadtxt(fp, skiprows=8)

# origin of data grid as stated in SRTM data file header
# create arrays with all lon/lat values from min to max and
#lats = np.linspace(-33,-17, srtm.shape[0])
#lons = np.linspace(-74, -60.0000, srtm.shape[1])

# create Basemap instance with Mercator projection
# we want a slightly smaller region than covered by our SRTM data



def ploteventstation(dir_observe,list_input):
 m = Basemap(projection='merc', lon_0=13, lat_0=48, resolution="h",llcrnrlon=-78, llcrnrlat=-36, urcrnrlon=-61, urcrnrlat=-12)

# create grids and compute map projection coordinates for lon/lat grid
#x, y = m(*np.meshgrid(lons, lats))

# Make contour plot
#cs = m.contourf(x, y, srtm,30,cmap=plt.cm.jet)
 m.drawcountries(color="blue", linewidth=0.1)
 m.shadedrelief()
 m.drawcoastlines(linewidth=0.5)
 m.drawcountries(linewidth=0.2)
 parallels = np.arange(-81,0,4.)
 m.drawparallels(parallels,linewidth=0,labels=[1,0,1,1])
 meridians = np.arange(10.,351.,4.)
 m.drawmeridians(meridians,linewidth=0,labels=[1,1,0,1])

 focmecs=[]
 lats=[]
 lons=[]
 stalons=np.array([])
 stalats=np.array([])
 files = glob.glob(dir_observe)
#files = glob.glob('*.h5')
 for f in files:
####### here implement the for circyle
       ds = pyasdf.ASDFDataSet(f)
       print(ds)
       event=ds.events[0]
       event_id=event.resource_id.id.split('=')[-1]
       magnitude=event.magnitudes[0].mag; Mtype=event.magnitudes[0].magnitude_type
       otime=event.origins[0].time
       evlo=event.origins[0].longitude; evla=event.origins[0].latitude; evdp=event.origins[0].depth/1000.
       #print(evlo)
       #print(evla)
       values=np.array([])
       valuetype='depth'
       mtensor=event.focal_mechanisms[0].moment_tensor.tensor
       mt=[mtensor.m_rr, mtensor.m_tt, mtensor.m_pp, mtensor.m_rt, mtensor.m_rp, mtensor.m_tp]
       focmecs.append(mt)
       lats.append(evla)
       lons.append(evlo)
       #print(focmecs)
       #print(lats,lons)
       x, y = m(lons, lats)
       #print(x,y)
       staLst=ds.waveforms.list()
       stalons=np.array([]); stalats=np.array([])
       for staid in list_input:
            stla, stlo,evz =ds.waveforms[staid].coordinates.values()
            print(stlo,stla)
            stalons=np.append(stalons, stlo)
            stalats=np.append(stalats, stla)
           
            #ax.add_collection(b)
            stax, stay=m(stalons, stalats)
            m.plot(stax, stay, '^', markersize=3)
        #m=self._get_basemap(projection=projection, geopolygons=geopolygons, blon=blon, blat=blat)
        #m.etopo()
        # m.shadedrelief()
        #stax, stay=m(stalons, stalats)
        #m.plot(stax, stay, '^', markersize=3)
        # plt.title(str(self.period)+' sec', fontsize=20)   
 ax = plt.gca()
# Two focal mechanisms for beachball routine, specified as [strike, dip, rake]
 for i in range(len(focmecs)):
       b = beach(focmecs[i], xy=(x[i], y[i]), width=100000, linewidth=0.2)
       b.set_zorder(10)
       ax.add_collection(b)
####################################################################

 plt.savefig('test.ps',format='ps')
 plt.close()




def plotevent_inv_station(dir_observe,list_input,m_inverse,shift_inlonlat,dir_output):
 m = Basemap(projection='merc', lon_0=13, lat_0=48, resolution="h",llcrnrlon=-78, llcrnrlat=-25, urcrnrlon=-61, urcrnrlat=-12)
 m.drawcountries(color="blue", linewidth=0.1)
 m.shadedrelief()
 m.drawcoastlines(linewidth=0.1)
 m.drawcountries(linewidth=0.1)
 parallels = np.arange(-81,0,4.)
 m.drawparallels(parallels,linewidth=0,labels=[1,0,1,1])
 meridians = np.arange(10.,351.,4.)
 m.drawmeridians(meridians,linewidth=0,labels=[1,1,0,1])

 focmecs=[]
 focmecs_inverse=m_inverse
 lats=[]
 lons=[]
 stalons=np.array([])
 stalats=np.array([])
 files = glob.glob(dir_observe)
#files = glob.glob('*.h5')
 for f in files:
####### here implement the for circyle
       ds = pyasdf.ASDFDataSet(f)
       print(ds)
       event=ds.events[0]
       event_id=event.resource_id.id.split('=')[-1]
       magnitude=event.magnitudes[0].mag; Mtype=event.magnitudes[0].magnitude_type
       otime=event.origins[0].time
       evlo=event.origins[0].longitude; evla=event.origins[0].latitude; evdp=event.origins[0].depth/1000.
       #print(evlo)
       #print(evla)
       values=np.array([])
       valuetype='depth'
       mtensor=event.focal_mechanisms[0].moment_tensor.tensor
       mt=[mtensor.m_rr, mtensor.m_tt, mtensor.m_pp, mtensor.m_rt, mtensor.m_rp, mtensor.m_tp]
       focmecs.append(mt)
       lats.append(evla)
       lons.append(evlo)
       #print(focmecs)
       #print(lats,lons)
       x, y = m(lons, lats)
       
       x1, y1=m(evlo+shift_inlonlat[0], evla+shift_inlonlat[1])
       #print(x,y)
       staLst=ds.waveforms.list()
       stalons=np.array([]); stalats=np.array([])
       for staid in list_input:
            stla, stlo,evz =ds.waveforms[staid].coordinates.values()
            print(stlo,stla)
            stalons=np.append(stalons, stlo)
            stalats=np.append(stalats, stla)
           
            #ax.add_collection(b)
            stax, stay=m(stalons, stalats)
            m.plot(stax, stay, 'b^', markersize=3)
        #m=self._get_basemap(projection=projection, geopolygons=geopolygons, blon=blon, blat=blat)
        #m.etopo()
        # m.shadedrelief()
        #stax, stay=m(stalons, stalats)
        #m.plot(stax, stay, '^', markersize=3)
        # plt.title(str(self.period)+' sec', fontsize=20)   
 ax = plt.gca()
# Two focal mechanisms for beachball routine, specified as [strike, dip, rake]
 for i in range(len(focmecs)):
       b = beach(focmecs[i], xy=(x[i], y[i]), width=100000, linewidth=0.2,facecolor='b')
       b.set_zorder(10)
       ax.add_collection(b)


 b1 = beach(focmecs_inverse, xy=(x1,y1), width=100000, linewidth=0.2,facecolor='r')
 b1.set_zorder(10)
 ax.add_collection(b1)

####################################################################

 plt.savefig(dir_output+'map.ps',format='ps')
 plt.close()
 return focmecs,focmecs_inverse



######this function is intended to plot the misfit_depth and focal mechanisms with depth change

def plot_misfit_depth_focal(misfit_array,depth_array,m_inverse_list,misfit_init,m_initial):
 ax = plt.gca()
 plt.plot(depth_array,misfit_array*200,'black')
 for _i in np.arange(0,depth_array.shape[0],1):
  #focmecs.append(m_inverse_list[_i])
   print(depth_array[_i],misfit_array[_i],m_inverse_list[_i])
   b = beach(m_inverse_list[_i], xy=(depth_array[_i],misfit_array[_i]*200),width=3,linewidth=0.2,facecolor='b')
   #b.set_zorder(10)
   ax.add_collection(b)
   ax.set_aspect("equal")
 b = beach(m_initial, xy=(0,misfit_init*200),width=3,linewidth=0.2,facecolor='r')
 ax.add_collection(b)
 ax.set_xlim((-35, 25))
 ax.set_ylim((220, 290)) 
 #xlabel=
 ax.grid()
 ax.set_ylabel('Misfit')
 ax.set_xlabel('Depth Perturpation(km)')
 plt.savefig(dir_output+'misfit_depth.ps',format='ps')
 plt.close()










