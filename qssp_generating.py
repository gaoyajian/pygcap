#!/usr/bin/env python
# -*- coding: utf-8 -*-
### this code is intended to generate the input files for qssp and read in the synthetics from the results of qssp and finally generate h5df files like salvus generated



############Mrr

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
#import functions

from obspy.taup import TauPyModel
import os, sys, glob, datetime


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import qsspIO



if len(sys.argv) != 4:
	sys.exit("Usage: python %s sir,please give me a component name such like 'Mrr' "% sys.argv[0])
component_name = [sys.argv[1]]
relative_depth=np.float(sys.argv[2])
dir=sys.argv[3]
print(component_name,relative_depth)



event_name='2867840'
dir_observe='2867840.h5'
component_name=component_name[0]
end_time=560
time_increment=0.07
fmin=1/80
fmax=1/5



list_input,sta,sto=qsspIO.retrieve_stations_latlon(dir_observe)
print(list_input)

#####here is the original information 
evla,evlo,evdp,m0,mrr,mtt,mpp,mrt,mrp,mtp,strike_nodal_1,dip_nodal_1,rake_nodal_1=qsspIO.retrieve_events_information(dir_observe)
M0,m_tt,m_pp,m_rp,m_rt,m_rr,m_tp=qsspIO.generate_mt_single_component_qssp(component_name,dir_observe,end_time,time_increment)

########here define a lot of variables which should be input to the qsspinputfile




# 1. uniform receiver depth [km]
#-------------------------------------------------------------------------------------------
receiver_depth=    0.00

#-------------------------------------------------------------------------------------------
#
#	TIME (FREQUENCY) SAMPLING
#	=========================
# 1. time window [sec], sampling interval [sec]
# 2. max. frequency [Hz] of Green's functions
# 3. max. slowness [s/km] of Green's functions
#    Note: if the near-field static displacement is desired, the maximum slowness should not
#          be smaller than the S wave slowness in the receiver layer
# 4. anti-aliasing factor (> 0 & < 1), if it is <= 0 or >= 1/e (~ 0.4), then
#    default value of 1/e is used (e.g., 0.1 = alias phases will be suppressed
#    to 10% of their original amplitude)
#
#    Note: The computation effort increases linearly the time window and
#          quadratically with the cut-off frequency.
#-------------------------------------------------------------------------------------------
time_window=   end_time   
sampling_interval=time_increment
max_freq=    1 ###this is the upper frequency for the computation
max_slow=0.5
anti_aliasing=1




# 1. the critical frequency [Hz] and the critical harmonic degree, below which
#    the self-gravitating effect should be included
#-------------------------------------------------------------------------------------------
cri_freq=0.0    
critical_harmonic_degree=0


# 1. selection (1/0 = yes/no) of speroidal modes (P-SV waves), selection of toroidal modes
#    (SH waves), minimum and maximum cutoff harmonic degrees
#    Note: if the near-field static displacement is desired, the minimum cutoff harmonic
#          degree should not be smaller than, e.g., 2000.
#-------------------------------------------------------------------------------------------
speroidal_modes=1     
toroidal_modes=1     
minimum_cutoff_harmonic_degrees=3000   
maximum_cutoff_harmonic_degrees=10000

#	GREEN'S FUNCTION FILES
#	======================
# 1. number of discrete source depths, estimated radius of each source patch [km] and
#    directory for Green's functions
# 2. list of the source depths [km], the respective file names of the Green's
#    functions (spectra) and the switch number (0/1) (0 = do not calculate
#    this Green's function because it exists already, 1 = calculate or update
#    this Green's function. Note: update is required if any of the above
#    parameters is changed)
#-------------------------------------------------------------------------------------------

number_discrete_sourcedepth=1
radius_sourcepach=1
directory_green= './GreenFunction/'
source_depth=evdp+relative_depth
green_function_name=component_name
green_update=1


#-------------------------------------------------------------------------------------------
#
#	MULTI-EVENT SOURCE PARAMETERS
#	=============================
# 1. number of discrete point sources and selection of the source data format
#    (1 or 2)
# 2. list of the multi-event sources
#    Format 1:
#    M-Unit   Mrr  Mtt  Mpp  Mrt  Mrp  Mtp  Lat   Lon   Depth  T_origin T_rise
#    [Nm]                                   [deg] [deg] [km]   [sec]    [sec]
#    Format 2:
#    Moment   Strike    Dip       Rake      Lat   Lon   Depth  T_origin T_rise
#    [Nm]     [deg]     [deg]     [deg]     [deg] [deg] [km]   [sec]    [sec]
#-------------------------------------------------------------------------------------------
number_pointsources=1
##########if it is the double couple source, it should be 1
selection_pointsource=1
M_Unit=M0
Mrr=m_rr
Mtt=m_tt
Mpp=m_pp
Mrt=m_rt
Mrp=m_rp
Mtp=m_tp
Lat=evla 
Lon  =evlo
######test Depth
Depth  =source_depth
T_origin =-time_increment
T_rise=0.2
# 1. output file name and and selection of output format:
#       1 = cartesian: vertical(z)/north(n)/east(e);
#       2 = spherical: radial(r)/theta(t)/phi(p)
#      (Note: if output format 2 is selected, the epicenter (T_origin = 0)
# 2. output time window [sec] (<= Green's function time window)
# 3. selection of order of Butterworth bandpass filter (if <= 0, then no filtering), lower
#    and upper corner frequencies (smaller than the cut-off frequency defined above)
# 4. lower and upper slowness cut-off [s/km] (slowness band-pass filter)
# 5. number of receiver
# 6. list of the station parameters
#    Format:
#    Lat     Lon    Name     Time_reduction
#    [deg]   [deg]           [sec]
#    (Note: Time_reduction = start time of the time window)
#-------------------------------------------------------------------------------------------
inputfile_name=dir+'/'+event_name+component_name+'.inp'
outputfilename=event_name+component_name
selection_coor=1
timewindow=end_time
order_butter=-1
bandpass_lower=fmin
bandpass_upper=fmax
slowness_lower=0.01   
slowness_upper=1

#####prem model
#qsspIO.qsspinputfile(inputfile_name,list_input,'prem_simple.nd',receiver_depth,time_window,sampling_interval,max_freq,max_slow,anti_aliasing,
#                  cri_freq, critical_harmonic_degree,speroidal_modes,toroidal_modes,minimum_cutoff_harmonic_degrees,   
#                  maximum_cutoff_harmonic_degrees,number_discrete_sourcedepth,radius_sourcepach,directory_green,source_depth,
#                  green_function_name,green_update,number_pointsources,selection_pointsource,M_Unit,Mrr,Mtt,Mpp,Mrt,Mrp,Mtp,
#                  Lat,Lon,Depth,T_origin,T_rise,outputfilename,selection_coor,timewindow,order_butter,bandpass_lower,
#                  bandpass_upper,slowness_lower,slowness_upper,sta,sto)


qsspIO.qsspinputfile(inputfile_name,list_input,'Wasjamodel.md',receiver_depth,time_window,sampling_interval,max_freq,max_slow,anti_aliasing,
                  cri_freq, critical_harmonic_degree,speroidal_modes,toroidal_modes,minimum_cutoff_harmonic_degrees,   
                  maximum_cutoff_harmonic_degrees,number_discrete_sourcedepth,radius_sourcepach,directory_green,source_depth,
                  green_function_name,green_update,number_pointsources,selection_pointsource,M_Unit,Mrr,Mtt,Mpp,Mrt,Mrp,Mtp,
                  Lat,Lon,Depth,T_origin,T_rise,outputfilename,selection_coor,timewindow,order_butter,bandpass_lower,
                  bandpass_upper,slowness_lower,slowness_upper,sta,sto)













