#!/usr/bin/env python
# -*- coding: utf-8 -*-
### this code is intended to generate the necessary time windows, station weights, Green functions matrix, observed data and necessary information for the source initial #information generating.

from __future__ import absolute_import

import copy
import pyasdf
import os,glob
import numpy as np
import pyasdf
#from obspy.signal.tf_misfits import plot_tf_gofs
from pyasdf import ASDFDataSet
import pyasdf
from scipy.signal import hilbert
import numpy as np
from obspy.signal.tf_misfit import plot_tf_misfits
from obspy.signal.tf_misfit import tpm
#from obspy.signal.tf_misfit import plot_tfr
from obspy.geodetics.base import gps2dist_azimuth
from obspy.taup import TauPyModel
import obspy
from obspy import Stream
from obspy.signal.cross_correlation import xcorr
from obspy.signal.cross_correlation import correlate
from obspy.signal.cross_correlation import xcorr_max
#from tf_misfit_parall import tfem
import numba
from multiprocessing import Pool
from time import sleep
import tqdm
import functions

from obspy.taup import TauPyModel
import os, sys, glob, datetime


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np



########ons manually and from a list of stations containing 3 component
def output_station_channel_window(asdf_dir,list_input,min_period,max_period,time_increment,before_firarrival,after_firarrival):
 windows=np.zeros((len(list_input),2),dtype=int)
 #print(windows.size)
 model = TauPyModel(model="iasp91")
 ds = pyasdf.ASDFDataSet(asdf_dir)
 event=ds.events[0]
 datetime =event.origins[0].time
 evlo=event.origins[0].longitude; 
 evla=event.origins[0].latitude; 
 evdp=event.origins[0].depth/1000.
 otime=event.origins[0].time
 #list=ds.waveforms.list()
 #print(list)
 #st=Stream()
 i=0
 for stid in list_input:

    stla,stlo,evz=ds.waveforms[stid].coordinates.values()
 ####here need to test the existance of the Z components
    #st1=ds.waveforms[stid].preprocessed_30s_to_80s
    dist,az,baz    =   gps2dist_azimuth(evla,evlo,stla,stlo)
    dt             =   time_increment
    fmin           =    1/max_period
    fmax           =    1/min_period
    arrivals = model.get_travel_times(source_depth_in_km=evdp,
                                  distance_in_degree=dist/111000)
    arrivals2 = model.get_travel_times_geo(source_depth_in_km=evdp,
                                  source_latitude_in_deg=evla,source_longitude_in_deg=evlo,
                                  receiver_latitude_in_deg=stla,receiver_longitude_in_deg=stlo)
    
    timewindow     =   int(arrivals[0].time/dt+after_firarrival/dt)
    timewindowLEFT =   int(arrivals[0].time/dt-before_firarrival/dt)
    windows[i][0]=timewindowLEFT
    windows[i][1] = timewindow
    i += 1
 return  windows

def output_station_channel_window_onlybodywave(asdf_dir,list_input,min_period,max_period,time_increment,before_firarrival,after_finarrival):
 windows=np.zeros((len(list_input),2),dtype=int)
 #print(windows.size)
 model = TauPyModel(model="iasp91")
 ds = pyasdf.ASDFDataSet(asdf_dir)
 event=ds.events[0]
 datetime =event.origins[0].time
 evlo=event.origins[0].longitude; 
 evla=event.origins[0].latitude; 
 evdp=event.origins[0].depth/1000.
 otime=event.origins[0].time
 #list=ds.waveforms.list()
 #print(list)
 #st=Stream()
 i=0
 for stid in list_input:

    stla,stlo,evz=ds.waveforms[stid].coordinates.values()
 ####here need to test the existance of the Z components
    #st1=ds.waveforms[stid].preprocessed_30s_to_80s
    dist,az,baz    =   gps2dist_azimuth(evla,evlo,stla,stlo)
    dt             =   time_increment
    fmin           =    1/max_period
    fmax           =    1/min_period
    arrivals = model.get_travel_times(source_depth_in_km=evdp,
                                  distance_in_degree=dist/111000)
    arrivals2 = model.get_travel_times_geo(source_depth_in_km=evdp,
                                  source_latitude_in_deg=evla,source_longitude_in_deg=evlo,
                                  receiver_latitude_in_deg=stla,receiver_longitude_in_deg=stlo)
    
    timewindow     =   int(arrivals[1].time/dt+after_finarrival/dt)
    timewindowLEFT =   int(arrivals[0].time/dt-before_firarrival/dt)
    windows[i][0]=timewindowLEFT
    windows[i][1] = timewindow
    i += 1
 return  windows




#this function is in order to generate the weights for the misfits according to the distance relative to the reference distance
# because most of the data is the surface wave, so the exponent should be 0.5
def output_station_weight(asdf_dir,list_input):
  weight=np.zeros(len(list_input),dtype=float)
  model = TauPyModel(model="iasp91")
  ds = pyasdf.ASDFDataSet(asdf_dir)
  event=ds.events[0]
  datetime =event.origins[0].time
  evlo=event.origins[0].longitude; 
  evla=event.origins[0].latitude; 
  evdp=event.origins[0].depth/1000.
  otime=event.origins[0].time
 #list=ds.waveforms.list()
 #print(list)
 #st=Stream()
  
  for _i,stid in enumerate(list_input):
    stla,stlo,evz=ds.waveforms[stid].coordinates.values()
    dist,az,baz    =   gps2dist_azimuth(evla,evlo,stla,stlo) ##km
    weight[_i]=np.sqrt(dist/200000)
  return weight














def Gmatrix_build_Z(dir_rr,dir_tt,dir_pp,dir_rt,dir_rp,dir_tp,list_input,windows):
 ###here read in the stations and channels and the window combined information list
 model = TauPyModel(model="iasp91")
 size=len(list_input)
 #print('station_list size is',size)
 length=np.zeros(size)
 for i in np.arange(0,size,1):
     length[i]=windows[i][1]-windows[i][0]+1
     #print(length[i])
     #length[i]=length[i]

 length=np.sum(length)
 #print(length)
 G = np.zeros((6, int(length)), dtype=float)

 ds_tt = pyasdf.ASDFDataSet(dir_tt)
 ds_pp = pyasdf.ASDFDataSet(dir_pp) 
 ds_rr = pyasdf.ASDFDataSet(dir_rr)
 ds_rp = pyasdf.ASDFDataSet(dir_rp)
 ds_rt = pyasdf.ASDFDataSet(dir_rt)
 ds_tp = pyasdf.ASDFDataSet(dir_tp)



 Z_index_left=0
 Z_index_right=0
 for _i,stid in enumerate(list_input):
   st_TT_A=ds_tt.waveforms[stid].displacement
   st_PP_A=ds_pp.waveforms[stid].displacement
   st_RP_A=ds_rp.waveforms[stid].displacement
   st_RT_A=ds_rt.waveforms[stid].displacement
   st_TP_A=ds_tp.waveforms[stid].displacement
   st_RR_A=ds_rr.waveforms[stid].displacement
   
#######here generate individual synthetics
   st_TT_ROTA_A_Z=st_TT_A.select(component="Z")[0].data
   #st_TT_ROTA_A_E=st_TT_A.select(component="E")[0].data
   #st_TT_ROTA_A_N=st_TT_A.select(component="N")[0].data

   st_PP_ROTA_A_Z=st_PP_A.select(component="Z")[0].data
   #st_PP_ROTA_A_E=st_PP_A.select(component="E")[0].data
   #st_PP_ROTA_A_N=st_PP_A.select(component="N")[0].data

   st_RP_ROTA_A_Z=st_RP_A.select(component="Z")[0].data
   #st_RP_ROTA_A_E=st_RP_A.select(component="E")[0].data
   #st_RP_ROTA_A_N=st_RP_A.select(component="N")[0].data

   st_RT_ROTA_A_Z=st_RT_A.select(component="Z")[0].data
   #st_RT_ROTA_A_E=st_RT_A.select(component="E")[0].data
   #st_RT_ROTA_A_N=st_RT_A.select(component="N")[0].data

   st_TP_ROTA_A_Z=st_TP_A.select(component="Z")[0].data
   #st_TP_ROTA_A_E=st_TP_A.select(component="E")[0].data
   #st_TP_ROTA_A_N=st_TP_A.select(component="N")[0].data


   st_RR_ROTA_A_Z=st_RR_A.select(component="Z")[0].data
   #st_RR_ROTA_A_E=st_RR_A.select(component="E")[0].data
   #st_RR_ROTA_A_N=st_RR_A.select(component="N")[0].data

   ##############
   
   
   Z_index_right+=windows[_i][1]-windows[_i][0]
   #print(windows[_i][1]-windows[_i][0])
   #print(Z_index_left,Z_index_right)
   G[0][Z_index_left:Z_index_right]=st_RR_ROTA_A_Z[windows[_i][0]:windows[_i][1]]
   G[1][Z_index_left:Z_index_right]=st_TT_ROTA_A_Z[windows[_i][0]:windows[_i][1]]
   G[2][Z_index_left:Z_index_right]=st_PP_ROTA_A_Z[windows[_i][0]:windows[_i][1]]
   G[3][Z_index_left:Z_index_right]=st_RT_ROTA_A_Z[windows[_i][0]:windows[_i][1]]
   G[4][Z_index_left:Z_index_right]=st_RP_ROTA_A_Z[windows[_i][0]:windows[_i][1]]
   G[5][Z_index_left:Z_index_right]=st_TP_ROTA_A_Z[windows[_i][0]:windows[_i][1]]
  
   Z_index_left+=windows[_i][1]-windows[_i][0]
   #print(Z_index_left)
 Gt=np.swapaxes(G,0,1)
 return Gt





def Gmatrix_build_E(dir_rr,dir_tt,dir_pp,dir_rt,dir_rp,dir_tp,list_input,windows):
 ###here read in the stations and channels and the window combined information list
 model = TauPyModel(model="iasp91")
 size=len(list_input)
 length=np.zeros(size)
 for i in np.arange(0,size,1):
     length[i]=windows[i][1]-windows[i][0]+1
     length[i]=length[i]

 length=np.sum(length)
 print(length)
 G = np.zeros((6, int(length)), dtype=float)

 ds_tt = pyasdf.ASDFDataSet(dir_tt)
 ds_pp = pyasdf.ASDFDataSet(dir_pp) 
 ds_rr = pyasdf.ASDFDataSet(dir_rr)
 ds_rp = pyasdf.ASDFDataSet(dir_rp)
 ds_rt = pyasdf.ASDFDataSet(dir_rt)
 ds_tp = pyasdf.ASDFDataSet(dir_tp)



 E_index_left=0
 E_index_right=0
 for _i,stid in enumerate(list_input):
   st_TT_A=ds_tt.waveforms[stid].displacement
   st_PP_A=ds_pp.waveforms[stid].displacement
   st_RP_A=ds_rp.waveforms[stid].displacement
   st_RT_A=ds_rt.waveforms[stid].displacement
   st_TP_A=ds_tp.waveforms[stid].displacement
   st_RR_A=ds_rr.waveforms[stid].displacement
   
#######here generate individual synthetics
   #st_TT_ROTA_A_Z=st_TT_A.select(component="Z")[0].data
   st_TT_ROTA_A_E=st_TT_A.select(component="E")[0].data
   #st_TT_ROTA_A_N=st_TT_A.select(component="N")[0].data

   #st_PP_ROTA_A_Z=st_PP_A.select(component="Z")[0].data
   st_PP_ROTA_A_E=st_PP_A.select(component="E")[0].data
   #st_PP_ROTA_A_N=st_PP_A.select(component="N")[0].data

   #st_RP_ROTA_A_Z=st_RP_A.select(component="Z")[0].data
   st_RP_ROTA_A_E=st_RP_A.select(component="E")[0].data
   #st_RP_ROTA_A_N=st_RP_A.select(component="N")[0].data

   #st_RT_ROTA_A_Z=st_RT_A.select(component="Z")[0].data
   st_RT_ROTA_A_E=st_RT_A.select(component="E")[0].data
   #st_RT_ROTA_A_N=st_RT_A.select(component="N")[0].data

   #st_TP_ROTA_A_Z=st_TP_A.select(component="Z")[0].data
   st_TP_ROTA_A_E=st_TP_A.select(component="E")[0].data
   #st_TP_ROTA_A_N=st_TP_A.select(component="N")[0].data


   #st_RR_ROTA_A_Z=st_RR_A.select(component="Z")[0].data
   st_RR_ROTA_A_E=st_RR_A.select(component="E")[0].data
   #st_RR_ROTA_A_N=st_RR_A.select(component="N")[0].data

   ##############
   
   
   E_index_right+=windows[_i][1]-windows[_i][0]
   
   G[0][E_index_left:E_index_right]=st_RR_ROTA_A_E[windows[_i][0]:windows[_i][1]]
   G[1][E_index_left:E_index_right]=st_TT_ROTA_A_E[windows[_i][0]:windows[_i][1]]
   G[2][E_index_left:E_index_right]=st_PP_ROTA_A_E[windows[_i][0]:windows[_i][1]]
   G[3][E_index_left:E_index_right]=st_RT_ROTA_A_E[windows[_i][0]:windows[_i][1]]
   G[4][E_index_left:E_index_right]=st_RP_ROTA_A_E[windows[_i][0]:windows[_i][1]]
   G[5][E_index_left:E_index_right]=st_TP_ROTA_A_E[windows[_i][0]:windows[_i][1]]
  
   E_index_left+=windows[_i][1]-windows[_i][0]
   print(E_index_left)
 Gt=np.swapaxes(G,0,1)
 return Gt









def Gmatrix_build_N(dir_rr,dir_tt,dir_pp,dir_rt,dir_rp,dir_tp,list_input,windows):
 ###here read in the stations and channels and the window combined information list
 model = TauPyModel(model="iasp91")
 size=len(list_input)
 length=np.zeros(size)
 for i in np.arange(0,size,1):
     length[i]=windows[i][1]-windows[i][0]+1
     length[i]=length[i]

 length=np.sum(length)
 print(length)
 G = np.zeros((6, int(length)), dtype=float)

 ds_tt = pyasdf.ASDFDataSet(dir_tt)
 ds_pp = pyasdf.ASDFDataSet(dir_pp) 
 ds_rr = pyasdf.ASDFDataSet(dir_rr)
 ds_rp = pyasdf.ASDFDataSet(dir_rp)
 ds_rt = pyasdf.ASDFDataSet(dir_rt)
 ds_tp = pyasdf.ASDFDataSet(dir_tp)



 N_index_left=0
 N_index_right=0
 for _i,stid in enumerate(list_input):
   st_TT_A=ds_tt.waveforms[stid].displacement
   st_PP_A=ds_pp.waveforms[stid].displacement
   st_RP_A=ds_rp.waveforms[stid].displacement
   st_RT_A=ds_rt.waveforms[stid].displacement
   st_TP_A=ds_tp.waveforms[stid].displacement
   st_RR_A=ds_rr.waveforms[stid].displacement
   
#######here generate individual synthetics
   #st_TT_ROTA_A_Z=st_TT_A.select(component="Z")[0].data
   #st_TT_ROTA_A_E=st_TT_A.select(component="E")[0].data
   st_TT_ROTA_A_N=st_TT_A.select(component="N")[0].data

   #st_PP_ROTA_A_Z=st_PP_A.select(component="Z")[0].data
   #st_PP_ROTA_A_E=st_PP_A.select(component="E")[0].data
   st_PP_ROTA_A_N=st_PP_A.select(component="N")[0].data

   #st_RP_ROTA_A_Z=st_RP_A.select(component="Z")[0].data
   #st_RP_ROTA_A_E=st_RP_A.select(component="E")[0].data
   st_RP_ROTA_A_N=st_RP_A.select(component="N")[0].data

   #st_RT_ROTA_A_Z=st_RT_A.select(component="Z")[0].data
   #st_RT_ROTA_A_E=st_RT_A.select(component="E")[0].data
   st_RT_ROTA_A_N=st_RT_A.select(component="N")[0].data

   #st_TP_ROTA_A_Z=st_TP_A.select(component="Z")[0].data
   #st_TP_ROTA_A_E=st_TP_A.select(component="E")[0].data
   st_TP_ROTA_A_N=st_TP_A.select(component="N")[0].data


   #st_RR_ROTA_A_Z=st_RR_A.select(component="Z")[0].data
   #st_RR_ROTA_A_E=st_RR_A.select(component="E")[0].data
   st_RR_ROTA_A_N=st_RR_A.select(component="N")[0].data

   ##############
   
   
   N_index_right+=windows[_i][1]-windows[_i][0]
   
   G[0][N_index_left:N_index_right]=st_RR_ROTA_A_N[windows[_i][0]:windows[_i][1]]
   G[1][N_index_left:N_index_right]=st_TT_ROTA_A_N[windows[_i][0]:windows[_i][1]]
   G[2][N_index_left:N_index_right]=st_PP_ROTA_A_N[windows[_i][0]:windows[_i][1]]
   G[3][N_index_left:N_index_right]=st_RT_ROTA_A_N[windows[_i][0]:windows[_i][1]]
   G[4][N_index_left:N_index_right]=st_RP_ROTA_A_N[windows[_i][0]:windows[_i][1]]
   G[5][N_index_left:N_index_right]=st_TP_ROTA_A_N[windows[_i][0]:windows[_i][1]]
  
   N_index_left+=windows[_i][1]-windows[_i][0]
   print(N_index_left)
 Gt=np.swapaxes(G,0,1)
 return Gt



#########at the same time, output a array of the index of the time window for the G and d


def d_build(dir_observe,list_input,windows,tagname):
 ds_obs = pyasdf.ASDFDataSet(dir_observe)
 size=len(list_input)
 length=np.zeros(size)
 for i in np.arange(0,size,1):
   length[i]=windows[i][1]-windows[i][0]+1
   length[i]=length[i]
 length=np.sum(length)
 print(length)
 d_Z = np.zeros((1, int(length)), dtype=float)
 d_E = np.zeros((1, int(length)), dtype=float)
 d_N = np.zeros((1, int(length)), dtype=float)
 N_index_left=0
 N_index_right=0
 index_left=np.array([],dtype=int)
 index_right=np.array([],dtype=int)
 for _i,stid in enumerate(list_input):
   st_obs_A=ds_obs.waveforms[stid][tagname]
   st_obs_ROTA_A_Z=st_obs_A.select(component="Z")[0].data
   st_obs_ROTA_A_E=st_obs_A.select(component="E")[0].data
   st_obs_ROTA_A_N=st_obs_A.select(component="N")[0].data
   N_index_right+=windows[_i][1]-windows[_i][0]
   index_left=np.append(index_left,N_index_left)
   index_right=np.append(index_right,N_index_right)
   d_Z[0][N_index_left:N_index_right]=st_obs_ROTA_A_Z[windows[_i][0]:windows[_i][1]]
   d_E[0][N_index_left:N_index_right]=st_obs_ROTA_A_E[windows[_i][0]:windows[_i][1]]
   d_N[0][N_index_left:N_index_right]=st_obs_ROTA_A_N[windows[_i][0]:windows[_i][1]]
   N_index_left+=windows[_i][1]-windows[_i][0]
   print(N_index_left)
 return d_Z,d_E,d_N,index_left,index_right




#################this function is intended to read in the synthetics from full moment tensor calculated from salvus or qssp

def d_syb_build(dir_direct,list_input,windows):
 ds_obs = pyasdf.ASDFDataSet(dir_direct)
 size=len(list_input)
 length=np.zeros(size)
 for i in np.arange(0,size,1):
   length[i]=windows[i][1]-windows[i][0]+1
   length[i]=length[i]
 length=np.sum(length)
 print(length)
 d_Z = np.zeros((1, int(length)), dtype=float)
 d_E = np.zeros((1, int(length)), dtype=float)
 d_N = np.zeros((1, int(length)), dtype=float)
 N_index_left=0
 N_index_right=0
 index_left=np.array([],dtype=int)
 index_right=np.array([],dtype=int)
 for _i,stid in enumerate(list_input):
   st_obs_A=ds_obs.waveforms[stid].displacement
   st_obs_ROTA_A_Z=st_obs_A.select(component="Z")[0].data
   st_obs_ROTA_A_E=st_obs_A.select(component="E")[0].data
   st_obs_ROTA_A_N=st_obs_A.select(component="N")[0].data
   N_index_right+=windows[i][1]-windows[i][0]
   index_left=np.append(index_left,N_index_left)
   index_right=np.append(index_right,N_index_right)
   d_Z[0][N_index_left:N_index_right]=st_obs_ROTA_A_Z[windows[i][0]:windows[i][1]]
   d_E[0][N_index_left:N_index_right]=st_obs_ROTA_A_E[windows[i][0]:windows[i][1]]
   d_N[0][N_index_left:N_index_right]=st_obs_ROTA_A_N[windows[i][0]:windows[i][1]]
   N_index_left+=windows[i][1]-windows[i][0]
   print(N_index_left)
 return d_Z,d_E,d_N,index_left,index_right









def retrieve_events_information(dir_observe):
 ds = pyasdf.ASDFDataSet(dir_observe)
 event=ds.events[0]
 datetime =event.origins[0].time
 evlo=event.origins[0].longitude; evla=event.origins[0].latitude; evdp=event.origins[0].depth/1000.
 otime=event.origins[0].time
 list=ds.waveforms.list()
 focal_mechanism=event.focal_mechanisms[0]['moment_tensor']
 m0=focal_mechanism['scalar_moment']
 mrr=focal_mechanism['tensor']['m_rr']/m0
 mtt=focal_mechanism['tensor']['m_tt']/m0
 mpp=focal_mechanism['tensor']['m_pp']/m0
 mrt=focal_mechanism['tensor']['m_rt']/m0
 mrp=focal_mechanism['tensor']['m_rp']/m0
 mtp=focal_mechanism['tensor']['m_tp']/m0 

 parameters1 = event.focal_mechanisms[0]['nodal_planes']['nodal_plane_1']
 strike_nodal_1=parameters1.strike
 dip_nodal_1=parameters1.dip
 rake_nodal_1=parameters1.rake
 m_rr_init,m_tt_init,m_pp_init,m_rp_init,m_rt_init,m_tp_init=functions.DC_MT(strike_nodal_1,dip_nodal_1,rake_nodal_1)
 m_init_fullMT=np.array([mrr,mtt,mpp,mrt,mrp,mtp])
 m_init_DC=np.array([m_rr_init,m_tt_init,m_pp_init,m_rt_init,m_rp_init,m_tp_init])
 return parameters1,m_init_fullMT,m_init_DC









########this function is intended to generate a list of station list which include three components only
def retrieve_stations_information(dir_observe,listname,tagname):
 with open(listname,"w") as myfile:
    focmecs=[]
    lats=[]
    lons=[]
    stalons=np.array([])
    stalats=np.array([])
    ds = pyasdf.ASDFDataSet(dir_observe)
    event=ds.events[0]
    datetime =event.origins[0].time
    evlo=event.origins[0].longitude; 
    evla=event.origins[0].latitude; 
    evdp=event.origins[0].depth/1000.
    otime=event.origins[0].time
    list=ds.waveforms.list()
    for stid in list:
     st=ds.waveforms[stid]
     if len(st[tagname])==3:
       stla,stlo,evz=ds.waveforms[stid].coordinates.values()
       dist,az,baz    =   gps2dist_azimuth(evla,evlo,stla,stlo)
       print (stid,stla,stlo,dist,az)
       myfile.write("%s %0.3f %0.3f %0.3f %0.3f\n" %(stid,stla,stlo,dist,az) ) 
 myfile.close()



#####this function intended to genertate a list of stations Uniformly distributed stations for inversion
def generate_stationslist_forsourceinversion(dir_observe,tagname):
 focmecs=[]
 lats=[]
 lons=[]
 stalons=np.array([])
 stalats=np.array([])
 ds = pyasdf.ASDFDataSet(dir_observe)
 event=ds.events[0]
 datetime =event.origins[0].time
 evlo=event.origins[0].longitude; 
 evla=event.origins[0].latitude; 
 evdp=event.origins[0].depth/1000.
 otime=event.origins[0].time
 list=ds.waveforms.list()
 selected_list=[]
 
 for _i,stid in enumerate(list):
  st=ds.waveforms[stid]
  #print(_i)
  #tracename='st'+tagname
  if len(st[tagname])==3:
    stla,stlo,evz=ds.waveforms[stid].coordinates.values() 
    dist,az,baz    =   gps2dist_azimuth(evla,evlo,stla,stlo)
    if dist/1000 > 100:
     if _i==0:
      selected_list.append(stid)
      stalons=np.append(stalons,stlo)
      stalats=np.append(stalats,stla)
      print(len(selected_list))
     else:
      dist_relat=np.array([])
      for j in np.arange(0,len(selected_list),1):   
       dist_relat_indi,az_relat,baz_relat    =   gps2dist_azimuth(stalats[j],stalons[j],stla,stlo)
       dist_relat=np.append(dist_relat,dist_relat_indi)
       #print(dist_relat)
      if np.min(dist_relat)/1000 < 100:
       print("abandon this station",stid)
      else:
         selected_list.append(stid)
         stalons=np.append(stalons,stlo)
         stalats=np.append(stalats,stla)     
         #print("select this station",stid)
 return selected_list  
  


























