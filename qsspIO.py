#!/usr/bin/env python
# -*- coding: utf-8 -*-
### this code is intended to generate the input files for qssp2016 and read in the synthetics from the results of qssp and finally generate h5df files like salvus generated

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
from obspy import Stream
from obspy.signal.cross_correlation import xcorr
from obspy.signal.cross_correlation import correlate
from obspy.signal.cross_correlation import xcorr_max
#from tf_misfit_parall import tfem
import numba
from multiprocessing import Pool
from time import sleep
import tqdm
#import functions

from obspy.taup import TauPyModel
import os, sys, glob, datetime


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


##############################################################
############################################################## input file generator for computing greens functions from qssp

def retrieve_event_infor(dir_observe):
 ds = pyasdf.ASDFDataSet(dir_observe)
 event=ds.events[0]
 origin=event.preferred_origin() or event.origins[0]
 datetime =origin.time
 evlo=origin.longitude; 
 evla=origin.latitude; 
 evdp=origin.depth/1000.
 otime=origin.time
 datetime =origin.time
 fc=event.preferred_focal_mechanism() or event.focal_mechanisms[0]
 focal_mechanism=fc['moment_tensor']
 m0=focal_mechanism['scalar_moment']
 mrr=focal_mechanism['tensor']['m_rr']/m0
 mtt=focal_mechanism['tensor']['m_tt']/m0
 mpp=focal_mechanism['tensor']['m_pp']/m0
 mrt=focal_mechanism['tensor']['m_rt']/m0
 mrp=focal_mechanism['tensor']['m_rp']/m0
 mtp=focal_mechanism['tensor']['m_tp']/m0 
 parameters1 = fc['nodal_planes']['nodal_plane_1']
 strike_nodal_1=parameters1.strike
 dip_nodal_1=parameters1.dip
 rake_nodal_1=parameters1.rake
 return evla,evlo,evdp,





def retrieve_events_information(dir_observe):
 ds = pyasdf.ASDFDataSet(dir_observe)
 event=ds.events[0]
 origin=event.preferred_origin() or event.origins[0]
 datetime =origin.time
 evlo=origin.longitude; 
 evla=origin.latitude; 
 evdp=origin.depth/1000.
 otime=origin.time
 datetime =origin.time
 fc=event.preferred_focal_mechanism() or event.focal_mechanisms[0]
 focal_mechanism=fc['moment_tensor']
 m0=focal_mechanism['scalar_moment']
 mrr=focal_mechanism['tensor']['m_rr']/m0
 mtt=focal_mechanism['tensor']['m_tt']/m0
 mpp=focal_mechanism['tensor']['m_pp']/m0
 mrt=focal_mechanism['tensor']['m_rt']/m0
 mrp=focal_mechanism['tensor']['m_rp']/m0
 mtp=focal_mechanism['tensor']['m_tp']/m0 
 parameters1 = fc['nodal_planes']['nodal_plane_1']
 strike_nodal_1=parameters1.strike
 dip_nodal_1=parameters1.dip
 rake_nodal_1=parameters1.rake
 return evla,evlo,evdp,m0,mrr,mtt,mpp,mrt,mrp,mtp,strike_nodal_1,dip_nodal_1,rake_nodal_1




def generate_mt_single_component_qssp(component_name,dir_observe,end_time,time_increment):
 ds = pyasdf.ASDFDataSet(dir_observe)
 event = ds.events[0]
 delta=time_increment
 start_time = -time_increment
 npts=int(round((end_time -start_time)/time_increment) + 1)
 fc=event.preferred_focal_mechanism() or event.focal_mechanisms[0]
 focal_mechanism=fc['moment_tensor']
 m0=focal_mechanism['scalar_moment'] 
 if component_name=='Mrr':
   M0=m0
   m_tt=0
   m_pp=0
   m_rp=0
   m_rt=0
   m_rr=np.sqrt(2)
   m_tp=0      
 elif component_name=='Mtt':
   M0=m0
   m_tt=np.sqrt(2)
   m_pp=0
   m_rp=0
   m_rt=0
   m_rr=0
   m_tp=0
 elif component_name=='Mpp':
   M0=m0
   m_tt=0
   m_pp=np.sqrt(2)
   m_rp=0
   m_rt=0
   m_rr=0
   m_tp=0  
 elif component_name=='Mtp':
   M0=m0
   m_tt=0
   m_pp=0
   m_rp=0
   m_rt=0
   m_rr=0
   m_tp=1
 elif component_name=='Mrt':
   M0=m0
   m_tt=0
   m_pp=0
   m_rp=0
   m_rt=1
   m_rr=0
   m_tp=0
 elif component_name=='Mrp':
   M0=m0
   m_tt=0
   m_pp=0
   m_rp=1
   m_rt=0
   m_rr=0
   m_tp=0
   
 elif component_name=='full':
   M0=m0
   m_rr=focal_mechanism['tensor']['m_rr']/m0
   m_tt=focal_mechanism['tensor']['m_tt']/m0
   m_pp=focal_mechanism['tensor']['m_pp']/m0
   m_rt=focal_mechanism['tensor']['m_rt']/m0
   m_rp=focal_mechanism['tensor']['m_rp']/m0
   m_tp=focal_mechanism['tensor']['m_tp']/m0 
 return M0,m_tt,m_pp,m_rp,m_rt,m_rr,m_tp 


def retrieve_stations_latlon(dir_observe):
 focmecs=[]
 lats=[]
 lons=[]
 station_list=[]
 sta=np.array([])
 sto=np.array([])
 ds = pyasdf.ASDFDataSet(dir_observe)
 event=ds.events[0]
 origin=event.preferred_origin() or event.origins[0] 
 datetime =origin.time
 evlo=origin.longitude; 
 evla=origin.latitude; 
 evdp=origin.depth/1000.
 otime=origin.time
 list=ds.waveforms.list()
 for stid in list:
    st=ds.waveforms[stid]
    stla,stlo,evz=ds.waveforms[stid].coordinates.values()
    sta=np.append(sta,stla)
    sto=np.append(sto,stlo)
    station_list.append(stid)
 return station_list,sta,sto













##########this function is intended to generate the necessary words for the explainations or something for the parameters
def qsspinputfile(inputfilename,list_input,model_name,receiver_depth,time_window,sampling_interval,max_freq,max_slow,anti_aliasing,
                  cri_freq,critical_harmonic_degree,speroidal_modes,toroidal_modes,minimum_cutoff_harmonic_degrees,
                  maximum_cutoff_harmonic_degrees,number_discrete_sourcedepth,radius_sourcepach,directory_green,
                  source_depth,green_function_name,green_update,number_pointsources,selection_pointsource,M_Unit,
                  Mrr,Mtt,Mpp,Mrt,Mrp,Mtp,Lat,Lon,Depth,T_origin,T_rise,outputfilename,selection_coor,timewindow,
                  order_butter,bandpass_lower,bandpass_upper,slowness_lower,slowness_upper,sta,sto):
 model=np.loadtxt(model_name)
 [m,n]=model.shape 
 with open(inputfilename,"w") as myfile:
    myfile.write("# This is the input file of FORTRAN77 program qssp2016 for calculating\n")
    myfile.write("# synthetic seismograms of a self-gravitating, spherically symmetric,\n")
    myfile.write("# isotropic and viscoelastic earth.\n")
    myfile.write("#\n")
    myfile.write("# by\n")
    myfile.write("# Rongjiang  Wang <wang@gfz-potsdam.de>\n")
    myfile.write("# Helmholtz-Centre Potsdam\n")
    myfile.write("# GFZ German Reseach Centre for Geosciences\n")
    myfile.write("# Telegrafenberg, D-14473 Potsdam, Germany\n")
    myfile.write("#\n")
    myfile.write("# Last modified: Potsdam, July, 2017\n")
    myfile.write("#\n")
    myfile.write("# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n")
    myfile.write("# If not specified, SI Unit System is used overall!\n")
    myfile.write("#\n")
    myfile.write("# Coordinate systems:\n")
    myfile.write("# spherical (r,t,p) with r = radial,\n")
    myfile.write("#                        t = co-latitude,\n")
    myfile.write("#                        p = east longitude.\n")
    myfile.write("# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n")
    myfile.write("# 1. uniform receiver depth [km]\n")
    ##########
    myfile.write("%0.3f\n" %(receiver_depth))
    ##########

    myfile.write("#	TIME (FREQUENCY) SAMPLING\n")
    myfile.write("# 1. time window [sec], sampling interval [sec]\n")
    myfile.write("# 2. max. frequency [Hz] of Green's functions\n")
    myfile.write("# 3. max. slowness [s/km] of Green's functions\n")
    myfile.write("#    Note: if the near-field static displacement is desired, the maximum slowness should not\n")
    myfile.write("#          be smaller than the S wave slowness in the receiver layer\n")
    myfile.write("# 4. anti-aliasing factor (> 0 & < 1), if it is <= 0 or >= 1/e (~ 0.4), then\n")
    myfile.write("#    default value of 1/e is used (e.g., 0.1 = alias phases will be suppressed\n")
    myfile.write("#    to 10% of their original amplitude)\n")
    myfile.write("# 5. switch (1/0 = yes/no) of turning-point filter, the range (d1, d2) of max. penetration\n")
    myfile.write("#    depth [km] (d1 is meaningless if it is smaller than the receiver/source depth, and\n")
    myfile.write("#    d2 is meaningless if it is equal to or larger than the earth radius)\n")
    myfile.write("#\n")
    myfile.write("#    Note: The turning-point filter (Line 5) works only for the extended QSSP code (e.g.,\n")
    myfile.write("#          qssp2016). if this filter is selected, all phases with the turning point\n")
    myfile.write("#          shallower than d1 or deeper than d2 will be filtered.\n")
    myfile.write("#\n")
    myfile.write("# 6. Earth radius [km], switch of free-surface-reflection filter (1/0 = with/without free\n")
    myfile.write("#    surface reflection)\n")
    myfile.write("#\n")
    myfile.write("#    Note: The free-surface-reflection filter (Line 6) works only for the extended QSSP\n")
    myfile.write("#          code (e.g., qssp2016). if this filter is selected, all phases with the turning\n")
    myfile.write("#          point shallower than d1 or deeper than d2 will be filtered.\n")
    myfile.write("#-------------------------------------------------------------------------------------------\n")
    myfile.write("#    Note: The computation effort increases linearly the time window and\n")
    myfile.write("#          quadratically with the cut-off frequency.\n")
    myfile.write("#-------------------------------------------------------------------------------------------\n")
   
    #########
    myfile.write("%0.3f %0.3f\n" %(time_window,sampling_interval))
    myfile.write("%0.3f\n" %(max_freq))
    myfile.write("%0.3f\n" %(max_slow))
    myfile.write("%0.3f\n" %(anti_aliasing))
    myfile.write("%i %0.3f %0.3f\n" %(0,2891,6371.0))
    myfile.write("%0.3f %i\n" %(6371.0,0))
    #########
    myfile.write("#-------------------------------------------------------------------------------------------\n")
    myfile.write("#\n")
    myfile.write("#	SELF-GRAVITATING EFFECT\n")
    myfile.write("#	=======================\n")
    myfile.write("# 1. the critical frequency [Hz] and the critical harmonic degree, below which\n")
    myfile.write("#    the self-gravitating effect should be included\n")
    myfile.write("#-------------------------------------------------------------------------------------------\n")
    #########
    myfile.write("%0.3f %i\n" %(cri_freq,critical_harmonic_degree))
    #########

    myfile.write("#-------------------------------------------------------------------------------------------\n")
    myfile.write("#\n")
    myfile.write("#	WAVE TYPES\n")
    myfile.write("#	==========\n")
    myfile.write("# 1. selection (1/0 = yes/no) of speroidal modes (P-SV waves), selection of toroidal modes\n")
    myfile.write("#    (SH waves), minimum and maximum cutoff harmonic degrees\n")
    myfile.write("#    Note: if the near-field static displacement is desired, the minimum cutoff harmonic\n")
    myfile.write("#          degree should not be smaller than, e.g., 2000.\n")
    myfile.write("#-------------------------------------------------------------------------------------------\n")
    ##########
    myfile.write("%i %i %i %i\n" %(speroidal_modes,toroidal_modes,minimum_cutoff_harmonic_degrees,maximum_cutoff_harmonic_degrees))
    ##########


    myfile.write("#-------------------------------------------------------------------------------------------\n")
    myfile.write("#	GREEN'S FUNCTION FILES\n")
    myfile.write("#	======================\n")
    myfile.write("# 1. number of discrete source depths, estimated radius of each source patch [km] and\n")
    myfile.write("#    directory for Green's functions\n")
    myfile.write("# 2. list of the source depths [km], the respective file names of the Green's\n")
    myfile.write("#    functions (spectra) and the switch number (0/1) (0 = do not calculate\n")
    myfile.write("#    this Green's function because it exists already, 1 = calculate or update\n")
    myfile.write("#    this Green's function. Note: update is required if any of the above\n")
    myfile.write("#    parameters is changed)\n")
    myfile.write("#-------------------------------------------------------------------------------------------\n")

    ##########
    myfile.write("%i %0.3f %s\n" %(number_discrete_sourcedepth,radius_sourcepach,directory_green))
    myfile.write("%0.3f %s %s\n" %(source_depth,green_function_name,green_update))
    ##########
    myfile.write("#-------------------------------------------------------------------------------------------\n")
    myfile.write("#\n")
    myfile.write("#	MULTI-EVENT SOURCE PARAMETERS\n")
    myfile.write("#	=============================\n")
    myfile.write("# 1. number of discrete point sources and selection of the source data format\n")
    myfile.write("#    (1 or 2)\n")
    myfile.write("# 2. list of the multi-event sources\n")
    myfile.write("#    Format 1:\n")
    myfile.write("#    M-Unit   Mrr  Mtt  Mpp  Mrt  Mrp  Mtp  Lat   Lon   Depth  T_origin T_rise\n")
    myfile.write("#    [Nm]                                   [deg] [deg] [km]   [sec]    [sec]\n")
    myfile.write("#    Format 2:\n")
    myfile.write("#    Moment   Strike    Dip       Rake      Lat   Lon   Depth  T_origin T_rise\n")
    myfile.write("#    [Nm]     [deg]     [deg]     [deg]     [deg] [deg] [km]   [sec]    [sec]\n")
    myfile.write("#-------------------------------------------------------------------------------------------\n")
    ##########

    myfile.write("%i %i\n" %(number_pointsources,selection_pointsource))
    myfile.write("%s %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f\n" %(M_Unit,Mrr,Mtt,Mpp,Mrt,Mrp,Mtp,Lat,Lon,Depth,T_origin,T_rise))
    ##########


    myfile.write("#-------------------------------------------------------------------------------------------\n")
    myfile.write("#\n")
    myfile.write("#	RECEIVER PARAMETERS\n")
    myfile.write("#	===================\n")
    myfile.write("# 1. selection of output observables (1/0 = yes/no)")
    myfile.write("# 2. output file name and and selection of output format:\n")
    myfile.write("#       1 = cartesian: vertical(z)/north(n)/east(e);\n")
    myfile.write("#       2 = spherical: radial(r)/theta(t)/phi(p)\n")
    myfile.write("#      (Note: if output format 2 is selected, the epicenter (T_origin = 0)\n")
    myfile.write("# 3. output time window [sec] (<= Green's function time window)\n")
    myfile.write("# 4. selection of order of Butterworth bandpass filter (if <= 0, then no filtering), lower\n")
    myfile.write("#    and upper corner frequencies (smaller than the cut-off frequency defined above)\n")
    myfile.write("# 5. lower and upper slowness cut-off [s/km] (slowness band-pass filter)\n")
    myfile.write("# 6. number of receiver\n")
    myfile.write("# 7. list of the station parameters\n")
    myfile.write("#    Format:\n")
    myfile.write("#    Lat     Lon    Name     Time_reduction\n")
    myfile.write("#    [deg]   [deg]           [sec]\n")
    myfile.write("#    (Note: Time_reduction = start time of the time window)\n")
    myfile.write("#-------------------------------------------------------------------------------------------\n")
    myfile.write("%i %i %i %i %i %i\n" %(1,0,0,0,0,0))
    myfile.write("%s %i\n" %(outputfilename,selection_coor))
    myfile.write("%0.3f\n" %(timewindow))
    myfile.write("%i %0.3f %0.3f\n" %(order_butter,bandpass_lower,bandpass_upper))
    myfile.write("%0.3f %0.3f\n" %(slowness_lower,slowness_upper))
    myfile.write("%i\n" %(len(list_input)))
    for _i in np.arange(0,len(list_input),1,dtype=int):
       myfile.write("%0.3f %0.3f %s %0.3f\n" %(sta[_i],sto[_i],list_input[_i],0))
       #return _i    
    #################model parameters
    myfile.write("#-------------------------------------------------------------------------------------------\n")
    myfile.write("#\n")
    myfile.write("#	                LAYERED EARTH MODEL (IASP91)\n")
    myfile.write("#                   ============================\n")
    myfile.write("# 1. number of data lines of the layered model and selection for including\n")
    myfile.write("#    the physical dispersion according Kamamori & Anderson (1977)\n")
    myfile.write("#-------------------------------------------------------------------------------------------\n")
    myfile.write("%i %i\n" %(m,0))    
    myfile.write("#-------------------------------------------------------------------------------------------\n")
    myfile.write("#\n")
    myfile.write("#	MULTILAYERED MODEL PARAMETERS (source site)\n")
    myfile.write("#	===========================================\n")
    myfile.write("# no   depth[km]    vp[km/s]    vs[km/s]    ro[g/cm^3]       qp       qs\n")
    myfile.write("#-------------------------------------------------------------------------------------------\n")
    #return(m)
    for _i in np.arange(0,m,1,dtype=int):
       myfile.write("%i %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f\n" %(_i+1,model[_i][0],model[_i][1],model[_i][2],model[_i][3],model[_i][4],model[_i][5])) 
       #return _i
    myfile.write("#---------------------------------end of all inputs-----------------------------------------\n")
    myfile.close()





##############################################################
############################################################## output file generator for computing greens functions from qssp
########build data container asdf and move them to the corresponding dir

##########this function is intended to do the process including the filter and taper like lasif do
def asdf_file_write_process(dir_observe,dir_synthetic,eventname,component,min_period,max_period):
  datafile_tmp=eventname+component
  min_period_dir=str(min_period)
  print(min_period_dir)
  datafile=dir_synthetic+'/'+eventname+component+'.uz'
  tag_name='preprocessed_'+ str(min_period) +'s_to_'+str(max_period)+'s'
  file=open(datafile)
  line=file.readlines()
  ####read in the stations 
  stations=line[0].rsplit()
  tmp=line[1:]
  data=np.zeros((len(tmp),len(stations)),dtype=float)
  for i in np.arange(0,len(tmp),1):
   data_tmp=tmp[i].rsplit()
   data_tmp=np.array(data_tmp,dtype=float)
   data[i][0:len(stations)]=data_tmp[0:len(stations)]
  data_Z=np.swapaxes(data,0,1)

  datafile=dir_synthetic+'/'+eventname+component+'.ue'
  file=open(datafile)
  line=file.readlines()
  ####read in the stations 
  stations=line[0].rsplit()
  tmp=line[1:]
  data=np.zeros((len(tmp),len(stations)),dtype=float)
  for i in np.arange(0,len(tmp),1):
   data_tmp=tmp[i].rsplit()
   data_tmp=np.array(data_tmp,dtype=float)
   data[i][0:len(stations)]=data_tmp[0:len(stations)]
  data_E=np.swapaxes(data,0,1)

  datafile=dir_synthetic+'/'+eventname+component+'.un'
  file=open(datafile)
  line=file.readlines()
  ####read in the stations 
  stations=line[0].rsplit()
  tmp=line[1:]
  data=np.zeros((len(tmp),len(stations)),dtype=float)
  for i in np.arange(0,len(tmp),1):
   data_tmp=tmp[i].rsplit()
   data_tmp=np.array(data_tmp,dtype=float)
   data[i][0:len(stations)]=data_tmp[0:len(stations)]
  data_N=np.swapaxes(data,0,1)
  if not os.path.exists(dir_synthetic+'/'+min_period_dir+'/'+eventname+component):
   os.makedirs(dir_synthetic+'/'+min_period_dir+'/'+eventname+component)
  receiver_dir=dir_synthetic+'/'+min_period_dir+'/'+eventname+component+'/receivers.h5'
  print(receiver_dir)
  if os.path.exists(receiver_dir):
   os.system('rm -rf'+' '+receiver_dir)
   os.system('cp'+' '+dir_observe+' '+receiver_dir)
  with pyasdf.ASDFDataSet(receiver_dir) as ds_syn:
   list=ds_syn.waveforms.list()
   for _i,stid in enumerate(list):    
    stla,stlo,evz=ds_syn.waveforms[stid].coordinates.values()
    st_syn_gre=ds_syn.waveforms[stid][tag_name]
    index_qssp_station=stations.index(stid)
    #st_syn_gre=ds_syn.waveforms[stid].displacement
    if len(st_syn_gre)==3:
      st_syn_gre.select(component="Z")[0].data=data_Z[index_qssp_station]
      st_syn_gre.select(component="N")[0].data=data_N[index_qssp_station]
      st_syn_gre.select(component="E")[0].data=data_E[index_qssp_station]
      st_syn_gre.detrend("linear")
      st_syn_gre.detrend("demean")
      st_syn_gre.taper(0.05, type="cosine")
      st_syn_gre.filter("bandpass", freqmin=1.0 / max_period,
                  freqmax=1.0 / min_period, corners=3, zerophase=False)
      st_syn_gre.detrend("linear")
      st_syn_gre.detrend("demean")
      st_syn_gre.taper(0.05, type="cosine")
      st_syn_gre.filter("bandpass", freqmin=1.0 / max_period,
                  freqmax=1.0 / min_period, corners=3, zerophase=False)
      ds_syn.add_waveforms(st_syn_gre,tag='displacement')
      del ds_syn.waveforms[stid][tag_name]
      del ds_syn.waveforms[stid].StationXML

####this function is intended to process observed data 



def obs_asdf_process(dir_observe,eventname,min_period,max_period):
  ##the name of the file of the preprocessed data name and directory
  processdir=eventname+'preprocessed'
  processdata='preprocessed_'+ str(min_period) +'s_to_'+str(max_period)+'s.h5'
  print(processdata)
  if os.path.exists(processdir):
   os.system('rm -rf'+' '+processdir)
   os.makedirs(processdir)
   os.system('cp'+' '+dir_observe+' '+processdir+'/'+processdata)
  with pyasdf.ASDFDataSet(processdir+'/'+processdata) as ds_obs:
   list=ds_obs.waveforms.list()
   tag_name='preprocessed_'+ str(min_period) +'s_to_'+str(max_period)+'s'
   for _i,stid in enumerate(list):    
    stla,stlo,evz=ds_obs.waveforms[stid].coordinates.values()
    st_obs=ds_obs.waveforms[stid].raw_recording
    #st_syn_gre=ds_syn.waveforms[stid].displacement
    st_obs.detrend("linear")
    st_obs.detrend("demean")
    st_obs.taper(0.05, type="cosine")
    st_obs.filter("bandpass", freqmin=1.0 / max_period,
                  freqmax=1.0 / min_period, corners=3, zerophase=False)
    st_obs.detrend("linear")
    st_obs.detrend("demean")
    st_obs.taper(0.05, type="cosine")
    st_obs.filter("bandpass", freqmin=1.0 / max_period,
                  freqmax=1.0 / min_period, corners=3, zerophase=False)
    
    ds_obs.add_waveforms(st_obs,tag=tag_name)
    del ds_obs.waveforms[stid].raw_recording
    del ds_obs.waveforms[stid].preprocess








import numpy as np
from lasif import LASIFError
from scipy import signal
from pyasdf import ASDFDataSet
from obspy.core import UTCDateTime


#######this function

def preprocessing_function_asdf(dir_obs,eventname,time_increment,end_time,min_period,max_period):
    processdir=eventname+'preprocessed'
    processdata='preprocessed_'+ str(min_period) +'s_to_'+str(max_period)+'s.h5'
    tag_name='preprocessed_'+ str(min_period) +'s_to_'+str(max_period)+'s'
    if os.path.exists(processdir+'/'+processdata):
      os.system('rm -rf'+' '+processdir+'/'+processdata)
     # os.makedirs(processdir)
      os.system('cp'+' '+dir_obs+' '+processdir+'/'+processdata)
    def zerophase_chebychev_lowpass_filter(trace, freqmax):
        """
        Custom Chebychev type two zerophase lowpass filter useful for
        decimation filtering.

        This filter is stable up to a reduction in frequency with a factor of
        10. If more reduction is desired, simply decimate in steps.

        Partly based on a filter in ObsPy.

        :param trace: The trace to be filtered.
        :param freqmax: The desired lowpass frequency.

        Will be replaced once ObsPy has a proper decimation filter.
        """
        # rp - maximum ripple of passband, rs - attenuation of stopband
        rp, rs, order = 1, 96, 1e99
        ws = freqmax / (trace.stats.sampling_rate * 0.5)  # stop band frequency
        wp = ws  # pass band frequency

        while True:
            if order <= 12:
                break
            wp *= 0.99
            order, wn = signal.cheb2ord(wp, ws, rp, rs, analog=0)

        b, a = signal.cheby2(order, rs, wn, btype="low", analog=0, output="ba")
        print(trace)
        # Apply twice to get rid of the phase distortion.
        trace.data = signal.filtfilt(b, a, trace.data)

    # =========================================================================
    # Read ASDF file
    # =========================================================================

    ds = pyasdf.ASDFDataSet(processdir+'/'+processdata) 
    print(ds)
    list=ds.waveforms.list()
    event = ds.events[0]
    dt=time_increment
    sampling_rate = 1.0 / dt    
    start_time = -time_increment
    npts=int(round((end_time -start_time)/time_increment) + 1)
    origin=event.preferred_origin() or event.origins[0]
    print(origin.time,start_time)
    start=UTCDateTime(origin.time)
    print(start+start_time)
    starttime=start_time + np.float(start)
    print(starttime)
    endtime = end_time+starttime
    duration = end_time -start_time

    f2 = 0.9 / max_period
    f3 = 1.1 / min_period
    # Recommendations from the SAC manual.
    f1 = 0.5 * f2
    f4 = 2.0 * f3
    pre_filt = (f1, f2, f3, f4)

    for _i,stid in enumerate(list): 
        #print(stid)   
        stla,stlo,evz=ds.waveforms[stid].coordinates.values()
        st=ds.waveforms[stid].raw_recording
        for tr in st:
           print(tr)
            # Trim to reduce processing costs
           #tr.trim(starttime - 0.2 * duration, endtime + 0.2 * duration)
           print(tr)
           while True:
                decimation_factor = int(dt /
                                        tr.stats.delta)
                # Decimate in steps for large sample rate reductions.
                if decimation_factor > 8:
                    decimation_factor = 8
                if decimation_factor > 1:
                    new_nyquist = tr.stats.sampling_rate / 2.0 / float(
                        decimation_factor)
                    #print(new_nyquist)
                    zerophase_chebychev_lowpass_filter(tr, new_nyquist)
                    print(tr)
                    tr.decimate(factor=decimation_factor, no_filter=True)
                else:
                    break
        inv=ds.waveforms[stid].StationXML
        # Detrend and taper
        #print('start')
        st.detrend("linear")
        st.detrend("demean")
        st.taper(max_percentage=0.05, type="hann")
        # Instrument correction
        try:
            st.attach_response(inv)
            st.remove_response(output="DISP", pre_filt=pre_filt,
                               zero_mean=False, taper=False)
        except Exception as e:
            net = inv.get_contents()['channels'][0].split('.', 2)[0]
            sta = inv.get_contents()['channels'][0].split('.', 2)[1]

            msg = ("Station: %s.%s could not be corrected with the help of"
                   " asdf file: '%s'. Due to: '%s'  Will be skipped.") \
                % (net, sta,
                   processing_info["asdf_input_filename"], e.__repr__()),
            raise LASIFError(msg)

        # Bandpass filtering
        st.detrend("linear")
        st.detrend("demean")
        st.taper(0.05, type="cosine")
        st.filter("bandpass", freqmin=1.0 / max_period,
                  freqmax=1.0 / min_period, corners=3, zerophase=False)

        st.detrend("linear")
        st.detrend("demean")
        st.taper(0.05, type="cosine")
        st.filter("bandpass", freqmin=1.0 / max_period,
                  freqmax=1.0 / min_period, corners=3, zerophase=False)

        # Sinc interpolation
        for tr in st:
            tr.data = np.require(tr.data, requirements="C")
        st.interpolate(sampling_rate=sampling_rate, method="lanczos",
                       starttime=starttime, window="blackman", a=12, npts=npts)
        # Convert to single precision to save space.
        for tr in st:
            tr.data = np.require(tr.data, dtype="float32", requirements="C")

        ds.add_waveforms(st,tag=tag_name)
        del ds.waveforms[stid].raw_recording
        del ds.waveforms[stid].preprocess


