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



if len(sys.argv) != 3:
	sys.exit("Usage: python %s sir,please give me a component name such like 'Mrr' "% sys.argv[0])
component_name = sys.argv[2:]
component_name=component_name[0]
dir_synthetic=sys.argv[1]
print(component_name,dir_synthetic)

dir_observe='2867840preprocessed/preprocessed_10s_to_80s.h5'
eventname='2867840'
qsspIO.asdf_file_write_process(dir_observe,dir_synthetic,eventname,component_name,10,80)














