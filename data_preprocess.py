#!/usr/bin/env python
# -*- coding: utf-8 -*-
### this code is intended to preprocess the data

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




dir_obs='2867840.h5'
eventname='2867840'
time_increment=0.07
end_time=560
min_period=10
max_period=80
qsspIO.preprocessing_function_asdf(dir_obs,eventname,time_increment,end_time,min_period,max_period)
















