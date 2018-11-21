#!/usr/bin/env python
# -*- coding: utf-8 -*-
### This code is intended to extract misfits and summations from the calculations calculated based on the functions defined in functions.py


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

##########


########################### no consideration of the amplitudes of the obs and syns

def syn_sum(strike,dip,rake,Gt_Z,Gt_E,Gt_N,d_z,d_e,d_n,index_left,index_right):
    m_rr, m_tt, m_pp, m_rp, m_rt, m_tp=functions.DC_MT(strike,dip,rake)
    m_tmp = np.array([m_rr, m_tt, m_pp, m_rt, m_rp, m_tp])
    #print(i, j, k)
    ### assemble the greens functions to synthetics
    d_syn_z = np.dot(Gt_Z, m_tmp)
    d_syn_e = np.dot(Gt_E, m_tmp)
    d_syn_n = np.dot(Gt_N, m_tmp)
    cc=0.
    for _i,stid in enumerate(list_input):
      cc1 = functions.L1_L2misfit(d_z[0][index_left[_i]:index_right[_i]], d_syn_z[index_left[_i]:index_right[_i]])
      cc2 = functions.L1_L2misfit(d_e[0][index_left[_i]:index_right[_i]], d_syn_e[index_left[_i]:index_right[_i]])
      cc3 = functions.L1_L2misfit(d_n[0][index_left[_i]:index_right[_i]], d_syn_n[index_left[_i]:index_right[_i]])
      cc+=cc1+cc2+cc3
    corr_value = cc/len(list_input)/3
    #print(corr_value)
    return corr_value



def syn_sum_amp(strike,dip,rake,Gt_Z,Gt_E,Gt_N,d_z,d_e,d_n,index_left,index_right):
    m_rr, m_tt, m_pp, m_rp, m_rt, m_tp=functions.DC_MT(strike,dip,rake)
    m_tmp = np.array([m_rr, m_tt, m_pp, m_rt, m_rp, m_tp])
    #print(i, j, k)
    ### assemble the greens functions to synthetics
    d_syn_z = np.dot(Gt_Z, m_tmp)
    d_syn_e = np.dot(Gt_E, m_tmp)
    d_syn_n = np.dot(Gt_N, m_tmp)
    cc=0.
    for _i,stid in enumerate(list_input):
      cc1 = functions.L1_L2misfit_amp(d_z[0][index_left[_i]:index_right[_i]], d_syn_z[index_left[_i]:index_right[_i]])
      cc2 = functions.L1_L2misfit_amp(d_e[0][index_left[_i]:index_right[_i]], d_syn_e[index_left[_i]:index_right[_i]])
      cc3 = functions.L1_L2misfit_amp(d_n[0][index_left[_i]:index_right[_i]], d_syn_n[index_left[_i]:index_right[_i]])
      cc+=cc1+cc2+cc3
    corr_value = cc/len(list_input)/3
    #print(corr_value)
    return corr_value


def syn_sum_amp_iso_clvd(parameter1,parameter2):
    Gt_Z=parameter1[0]
    Gt_E=parameter1[1]
    Gt_N=parameter1[2]
    d_z=parameter1[3]
    d_e=parameter1[4]
    d_n=parameter1[5]
    index_left=parameter1[6]
    index_right=parameter1[7]
    list_input=parameter1[8]
    MTparameter=parameter2
    m_rr, m_tt, m_pp, m_rp, m_rt, m_tp=functions.nmtensor(MTparameter[0],MTparameter[1],MTparameter[2],MTparameter[3],MTparameter[4])
    m_tmp = np.array([m_rr, m_tt, m_pp, m_rt, m_rp, m_tp])
    #print(i, j, k)
    ### assemble the greens functions to synthetics
    d_syn_z = np.dot(Gt_Z, m_tmp)
    d_syn_e = np.dot(Gt_E, m_tmp)
    d_syn_n = np.dot(Gt_N, m_tmp)
    cc=0.
    for _i,stid in enumerate(list_input):
      cc1 = functions.L1_L2misfit_amp(d_z[0][index_left[_i]:index_right[_i]], d_syn_z[index_left[_i]:index_right[_i]])
      cc2 = functions.L1_L2misfit_amp(d_e[0][index_left[_i]:index_right[_i]], d_syn_e[index_left[_i]:index_right[_i]])
      cc3 = functions.L1_L2misfit_amp(d_n[0][index_left[_i]:index_right[_i]], d_syn_n[index_left[_i]:index_right[_i]])
      cc+=cc1+cc2+cc3
    corr_value = cc/len(list_input)/3
    #print(corr_value)
    return corr_value


def syn_sum_amp_iso_clvd_weight(parameter1,parameter2):
    Gt_Z=parameter1[0]
    Gt_E=parameter1[1]
    Gt_N=parameter1[2]
    d_z=parameter1[3]
    d_e=parameter1[4]
    d_n=parameter1[5]
    weight=parameter1[9]
    index_left=parameter1[6]
    index_right=parameter1[7]
    list_input=parameter1[8]
    MTparameter=parameter2
    m_rr, m_tt, m_pp, m_rp, m_rt, m_tp=functions.nmtensor(MTparameter[0],MTparameter[1],MTparameter[2],MTparameter[3],MTparameter[4])
    m_tmp = np.array([m_rr, m_tt, m_pp, m_rt, m_rp, m_tp])
    #print(i, j, k)
    ### assemble the greens functions to synthetics
    d_syn_z = np.dot(Gt_Z, m_tmp)
    d_syn_e = np.dot(Gt_E, m_tmp)
    d_syn_n = np.dot(Gt_N, m_tmp)
    cc=0.
    for _i,stid in enumerate(list_input):
      cc1 = functions.L1_L2misfit_amp(d_z[0][index_left[_i]:index_right[_i]], d_syn_z[index_left[_i]:index_right[_i]])
      cc2 = functions.L1_L2misfit_amp(d_e[0][index_left[_i]:index_right[_i]], d_syn_e[index_left[_i]:index_right[_i]])
      cc3 = functions.L1_L2misfit_amp(d_n[0][index_left[_i]:index_right[_i]], d_syn_n[index_left[_i]:index_right[_i]])
      cc+=weight[_i]*(cc1+cc2+cc3)
    corr_value = cc/len(list_input)/3
    #print(corr_value)
    return corr_value




######take the consideration of the relative M0 
def syn_sum_amp_iso_clvd_weight_M0(parameter1,parameter2):
    Gt_Z=parameter1[0]
    Gt_E=parameter1[1]
    Gt_N=parameter1[2]
    d_z=parameter1[3]
    d_e=parameter1[4]
    d_n=parameter1[5]
    weight=parameter1[9]
    index_left=parameter1[6]
    index_right=parameter1[7]
    list_input=parameter1[8]
    MTparameter=parameter2
    m_rr, m_tt, m_pp, m_rp, m_rt, m_tp=functions.nmtensor(MTparameter[0],MTparameter[1],MTparameter[2],MTparameter[3],MTparameter[4])
    m_tmp = MTparameter[5]*np.array([m_rr, m_tt, m_pp, m_rt, m_rp, m_tp])
    #print(i, j, k)
    ### assemble the greens functions to synthetics
    d_syn_z = np.dot(Gt_Z, m_tmp)
    d_syn_e = np.dot(Gt_E, m_tmp)
    d_syn_n = np.dot(Gt_N, m_tmp)
    cc=0.
    for _i,stid in enumerate(list_input):
      cc1 = functions.L1_L2misfit_amp(d_z[0][index_left[_i]:index_right[_i]], d_syn_z[index_left[_i]:index_right[_i]])
      cc2 = functions.L1_L2misfit_amp(d_e[0][index_left[_i]:index_right[_i]], d_syn_e[index_left[_i]:index_right[_i]])
      cc3 = functions.L1_L2misfit_amp(d_n[0][index_left[_i]:index_right[_i]], d_syn_n[index_left[_i]:index_right[_i]])
      cc+=weight[_i]*(cc1+cc2+cc3)
    corr_value = cc/len(list_input)/3
    #print(corr_value)
    return corr_value


## take consider the functions called L1_L2misfit_amp_deshifteff


def syn_sum_amp_iso_clvd_weight_M0_deshifteff(parameter1,parameter2):
    Gt_Z=parameter1[0]
    Gt_E=parameter1[1]
    Gt_N=parameter1[2]
    d_z=parameter1[3]
    d_e=parameter1[4]
    d_n=parameter1[5]
    weight=parameter1[9]
    index_left=parameter1[6]
    index_right=parameter1[7]
    list_input=parameter1[8]
    MTparameter=parameter2
    m_rr, m_tt, m_pp, m_rp, m_rt, m_tp=functions.nmtensor(MTparameter[0],MTparameter[1],MTparameter[2],MTparameter[3],MTparameter[4])
    m_tmp = MTparameter[5]*np.array([m_rr, m_tt, m_pp, m_rt, m_rp, m_tp])
    #print(i, j, k)
    ### assemble the greens functions to synthetics
    d_syn_z = np.dot(Gt_Z, m_tmp)
    d_syn_e = np.dot(Gt_E, m_tmp)
    d_syn_n = np.dot(Gt_N, m_tmp)
    cc=0.
    for _i,stid in enumerate(list_input):
      cc1 = functions.L1_L2misfit_amp_deshifteff(d_z[0][index_left[_i]:index_right[_i]], d_syn_z[index_left[_i]:index_right[_i]])
      cc2 = functions.L1_L2misfit_amp_deshifteff(d_e[0][index_left[_i]:index_right[_i]], d_syn_e[index_left[_i]:index_right[_i]])
      cc3 = functions.L1_L2misfit_amp_deshifteff(d_n[0][index_left[_i]:index_right[_i]], d_syn_n[index_left[_i]:index_right[_i]])
      cc+=weight[_i]*(cc1+cc2+cc3)
    corr_value = cc/len(list_input)/3
    #print(corr_value)
    return corr_value






def syn_sum_amp_iso_clvd_weight_M0_CC_body(parameter1,parameter2):
    Gt_Z=parameter1[0]
    Gt_E=parameter1[1]
    Gt_N=parameter1[2]
    d_z=parameter1[3]
    d_e=parameter1[4]
    d_n=parameter1[5]
    weight=parameter1[9]
    index_left=parameter1[6]
    index_right=parameter1[7]
    list_input=parameter1[8]
    MTparameter=parameter2
    m_rr, m_tt, m_pp, m_rp, m_rt, m_tp=functions.nmtensor(MTparameter[0],MTparameter[1],MTparameter[2],MTparameter[3],MTparameter[4])
    m_tmp = MTparameter[5]*np.array([m_rr, m_tt, m_pp, m_rt, m_rp, m_tp])
    #print(i, j, k)
    ### assemble the greens functions to synthetics
    d_syn_z = np.dot(Gt_Z, m_tmp)
    d_syn_e = np.dot(Gt_E, m_tmp)
    d_syn_n = np.dot(Gt_N, m_tmp)
    cc=0.
    for _i,stid in enumerate(list_input):
      cc1 = functions.cc_amp(d_z[0][index_left[_i]:index_right[_i]], d_syn_z[index_left[_i]:index_right[_i]])
      cc2 = functions.cc_amp(d_e[0][index_left[_i]:index_right[_i]], d_syn_e[index_left[_i]:index_right[_i]])
      cc3 = functions.cc_amp(d_n[0][index_left[_i]:index_right[_i]], d_syn_n[index_left[_i]:index_right[_i]])
      cc+=weight[_i]*(cc1+cc2+cc3)
    corr_value = cc/len(list_input)/3
    #print(corr_value)
    return corr_value





def syn_sum_amp_iso_clvd_weight_M0_CC_surface(parameter1,parameter2):
    Gt_Z=parameter1[0]
    Gt_E=parameter1[1]
    Gt_N=parameter1[2]
    d_z=parameter1[3]
    d_e=parameter1[4]
    d_n=parameter1[5]
    weight=parameter1[9]
    index_left=parameter1[6]
    index_right=parameter1[7]
    list_input=parameter1[8]
    MTparameter=parameter2
    m_rr, m_tt, m_pp, m_rp, m_rt, m_tp=functions.nmtensor(MTparameter[0],MTparameter[1],MTparameter[2],MTparameter[3],MTparameter[4])
    m_tmp = MTparameter[5]*np.array([m_rr, m_tt, m_pp, m_rt, m_rp, m_tp])
    #print(i, j, k)
    ### assemble the greens functions to synthetics
    d_syn_z = np.dot(Gt_Z, m_tmp)
    d_syn_e = np.dot(Gt_E, m_tmp)
    d_syn_n = np.dot(Gt_N, m_tmp)
    cc=0.
    for _i,stid in enumerate(list_input):
      cc1 = functions.cc_amp(d_z[0][index_left[_i]:index_right[_i]], d_syn_z[index_left[_i]:index_right[_i]])
      cc2 = functions.cc_amp(d_e[0][index_left[_i]:index_right[_i]], d_syn_e[index_left[_i]:index_right[_i]])
      cc3 = functions.cc_amp(d_n[0][index_left[_i]:index_right[_i]], d_syn_n[index_left[_i]:index_right[_i]])
      cc+=np.sqrt(weight[_i])*(cc1+cc2+cc3)
    corr_value = cc/len(list_input)/3
    #print(corr_value)
    return corr_value












######this function is intended to test the L2 from Lupei Zhu's Lushan paper
def syn_sum_L2_iso_clvd_weight(parameter1,parameter2):
    Gt_Z=parameter1[0]
    Gt_E=parameter1[1]
    Gt_N=parameter1[2]
    d_z=parameter1[3]
    d_e=parameter1[4]
    d_n=parameter1[5]
    weight=parameter1[9]
    index_left=parameter1[6]
    index_right=parameter1[7]
    list_input=parameter1[8]
    MTparameter=parameter2
    m_rr, m_tt, m_pp, m_rp, m_rt, m_tp=functions.nmtensor(MTparameter[0],MTparameter[1],MTparameter[2],MTparameter[3],MTparameter[4])
    m_tmp = np.array([m_rr, m_tt, m_pp, m_rt, m_rp, m_tp])
    #print(i, j, k)
    ### assemble the greens functions to synthetics
    d_syn_z = np.dot(Gt_Z, m_tmp)
    d_syn_e = np.dot(Gt_E, m_tmp)
    d_syn_n = np.dot(Gt_N, m_tmp)
    cc=0.
    for _i,stid in enumerate(list_input):
      cc1 = functions.L2misfit(d_z[0][index_left[_i]:index_right[_i]], d_syn_z[index_left[_i]:index_right[_i]])
      cc2 = functions.L2misfit(d_e[0][index_left[_i]:index_right[_i]], d_syn_e[index_left[_i]:index_right[_i]])
      cc3 = functions.L2misfit(d_n[0][index_left[_i]:index_right[_i]], d_syn_n[index_left[_i]:index_right[_i]])
      cc+=weight[_i]*(cc1+cc2+cc3)
    corr_value = cc/len(list_input)/3
    #print(corr_value)
    return corr_value


######this function is intended to test the L2 from Lupei Zhu's San Jacino paper which calculate the L2 directly without no normalise the amplitude
def syn_sum_L2_iso_clvd_weight_abs(parameter1,parameter2):
    Gt_Z=parameter1[0]
    Gt_E=parameter1[1]
    Gt_N=parameter1[2]
    d_z=parameter1[3]
    d_e=parameter1[4]
    d_n=parameter1[5]
    weight=parameter1[9]
    index_left=parameter1[6]
    index_right=parameter1[7]
    list_input=parameter1[8]
    MTparameter=parameter2
    m_rr, m_tt, m_pp, m_rp, m_rt, m_tp=functions.nmtensor(MTparameter[0],MTparameter[1],MTparameter[2],MTparameter[3],MTparameter[4])
    m_tmp = np.array([m_rr, m_tt, m_pp, m_rt, m_rp, m_tp])
    #print(i, j, k)
    ### assemble the greens functions to synthetics
    d_syn_z = np.dot(Gt_Z, m_tmp)
    d_syn_e = np.dot(Gt_E, m_tmp)
    d_syn_n = np.dot(Gt_N, m_tmp)
    cc=0.
    for _i,stid in enumerate(list_input):
      cc1 = functions.L2misfit_abs(d_z[0][index_left[_i]:index_right[_i]], d_syn_z[index_left[_i]:index_right[_i]])
      cc2 = functions.L2misfit_abs(d_e[0][index_left[_i]:index_right[_i]], d_syn_e[index_left[_i]:index_right[_i]])
      cc3 = functions.L2misfit_abs(d_n[0][index_left[_i]:index_right[_i]], d_syn_n[index_left[_i]:index_right[_i]])
      cc+=weight[_i]*weight[_i]*(cc1+cc2+cc3)
    corr_value = cc/len(list_input)/3
    #print(corr_value)
    return corr_value





######this function is intended to test the L2 and Mo inversion from Lupei Zhu's San Jacino paper which calculate the L2 directly without no normalise the amplitude
def syn_sum_L2_iso_clvd_weight_abs_M0(parameter1,parameter2):
    Gt_Z=parameter1[0]
    Gt_E=parameter1[1]
    Gt_N=parameter1[2]
    d_z=parameter1[3]
    d_e=parameter1[4]
    d_n=parameter1[5]
    weight=parameter1[9]
    index_left=parameter1[6]
    index_right=parameter1[7]
    list_input=parameter1[8]
    MTparameter=parameter2
    m_rr, m_tt, m_pp, m_rp, m_rt, m_tp=functions.nmtensor(MTparameter[0],MTparameter[1],MTparameter[2],MTparameter[3],MTparameter[4])
    #### here I inserted a amplification for the relative moment scalar for inverse the M0
    m_tmp = MTparameter[5]*np.array([m_rr, m_tt, m_pp, m_rt, m_rp, m_tp])    #print(i, j, k)
    ### assemble the greens functions to synthetics
    d_syn_z = np.dot(Gt_Z, m_tmp)
    d_syn_e = np.dot(Gt_E, m_tmp)
    d_syn_n = np.dot(Gt_N, m_tmp)
    cc=0.
    for _i,stid in enumerate(list_input):
      cc1 = functions.L2misfit_abs(d_z[0][index_left[_i]:index_right[_i]], d_syn_z[index_left[_i]:index_right[_i]])
      cc2 = functions.L2misfit_abs(d_e[0][index_left[_i]:index_right[_i]], d_syn_e[index_left[_i]:index_right[_i]])
      cc3 = functions.L2misfit_abs(d_n[0][index_left[_i]:index_right[_i]], d_syn_n[index_left[_i]:index_right[_i]])
      cc+=weight[_i]*weight[_i]*(cc1+cc2+cc3)
    corr_value = cc/len(list_input)/3
    #print(corr_value)
    return corr_value









######this function is intended to test the L2 from Lupei Zhu's Lushan paper, take the ratio amplitudes into consideration
def syn_sum_L2_iso_clvd_weight_amp(parameter1,parameter2):
    Gt_Z=parameter1[0]
    Gt_E=parameter1[1]
    Gt_N=parameter1[2]
    d_z=parameter1[3]
    d_e=parameter1[4]
    d_n=parameter1[5]
    weight=parameter1[9]
    index_left=parameter1[6]
    index_right=parameter1[7]
    list_input=parameter1[8]
    MTparameter=parameter2
    m_rr, m_tt, m_pp, m_rp, m_rt, m_tp=functions.nmtensor(MTparameter[0],MTparameter[1],MTparameter[2],MTparameter[3],MTparameter[4])
    m_tmp = np.array([m_rr, m_tt, m_pp, m_rt, m_rp, m_tp])
    #print(i, j, k)
    ### assemble the greens functions to synthetics
    d_syn_z = np.dot(Gt_Z, m_tmp)
    d_syn_e = np.dot(Gt_E, m_tmp)
    d_syn_n = np.dot(Gt_N, m_tmp)
    cc=0.
    for _i,stid in enumerate(list_input):
      cc1 = functions.L2misfit_amp(d_z[0][index_left[_i]:index_right[_i]], d_syn_z[index_left[_i]:index_right[_i]])
      cc2 = functions.L2misfit_amp(d_e[0][index_left[_i]:index_right[_i]], d_syn_e[index_left[_i]:index_right[_i]])
      cc3 = functions.L2misfit_amp(d_n[0][index_left[_i]:index_right[_i]], d_syn_n[index_left[_i]:index_right[_i]])
      print(cc)
      cc+=weight[_i]*(cc1+cc2+cc3)
    print(cc)
    corr_value = cc/len(list_input)/3
    #print(corr_value)
    return corr_value








def syn_sum_CC(parameter1,parameter2):
    Gt_Z=parameter1[0]
    Gt_E=parameter1[1]
    Gt_N=parameter1[2]
    d_z=parameter1[3]
    d_e=parameter1[4]
    d_n=parameter1[5]
    weight=parameter1[9]
    index_left=parameter1[6]
    index_right=parameter1[7]
    list_input=parameter1[8]
    MTparameter=parameter2
    m_rr, m_tt, m_pp, m_rp, m_rt, m_tp=functions.nmtensor(MTparameter[0],MTparameter[1],MTparameter[2],MTparameter[3],MTparameter[4])
    m_tmp = np.array([m_rr, m_tt, m_pp, m_rt, m_rp, m_tp])
    #print(i, j, k)
    ### assemble the greens functions to synthetics
    d_syn_z = np.dot(Gt_Z, m_tmp)
    d_syn_e = np.dot(Gt_E, m_tmp)
    d_syn_n = np.dot(Gt_N, m_tmp)
    cc=0.
    for _i,stid in enumerate(list_input):
      shift_z,cc1 = functions.corrNEW_amp(d_z[0][index_left[_i]:index_right[_i]], d_syn_z[index_left[_i]:index_right[_i]])
      shift_z,cc2 = functions.corrNEW_amp(d_e[0][index_left[_i]:index_right[_i]], d_syn_e[index_left[_i]:index_right[_i]])
      shift_z,cc3 = functions.corrNEW_amp(d_n[0][index_left[_i]:index_right[_i]], d_syn_n[index_left[_i]:index_right[_i]])
      cc+=weight[_i]*(cc1+cc2+cc3)
    corr_value = cc/len(list_input)/3
    return corr_value



def syn_sum_fem_fpm(strike,dip,rake,iso,clvd,Gt_Z,Gt_E,Gt_N,d_z,d_e,d_n,index_left,index_right):
    m_rr, m_tt, m_pp, m_rp, m_rt, m_tp=functions.nmtensor(strike,dip,rake)
    m_tmp = np.array([m_rr, m_tt, m_pp, m_rt, m_rp, m_tp])
    #print(i, j, k)
    ### assemble the greens functions to synthetics
    d_syn_z = np.dot(Gt_Z, m_tmp)
    d_syn_e = np.dot(Gt_E, m_tmp)
    d_syn_n = np.dot(Gt_N, m_tmp)
    cc=0.
    for _i,stid in enumerate(list_input):
      cc1 = functions.TF_fem(d_z[0][index_left[_i]:index_right[_i]], d_syn_z[index_left[_i]:index_right[_i]])
      cc2 = functions.TF_fem(d_e[0][index_left[_i]:index_right[_i]], d_syn_e[index_left[_i]:index_right[_i]])
      cc3 = functions.TF_fem(d_n[0][index_left[_i]:index_right[_i]], d_syn_n[index_left[_i]:index_right[_i]])
      cc1+= functions.TF_fpm(d_z[0][index_left[_i]:index_right[_i]], d_syn_z[index_left[_i]:index_right[_i]])
      cc2+= functions.TF_fpm(d_e[0][index_left[_i]:index_right[_i]], d_syn_e[index_left[_i]:index_right[_i]])
      cc3+= functions.TF_fpm(d_n[0][index_left[_i]:index_right[_i]], d_syn_n[index_left[_i]:index_right[_i]])
      cc+=cc1+cc2+cc3
    corr_value = cc/len(list_input)/3
    return corr_value




def calculate_misfits(strike_store,dip_store,rake_store,iso_store,clvd_store,Gt_Z,Gt_E,Gt_N,d_z,d_e,d_n,index_left,index_right):
    from tqdm import tqdm
    import multiprocessing as mp 
    pool = mp.Pool(processes=8)
    print(mp.cpu_count())
    parameter=[[strike_store[i],dip_store[j],rake_store[k],iso_store[m],clvd_store[n],Gt_Z,Gt_E,Gt_N,d_z,d_e,d_n,index_left,index_right]
        for m in tqdm(np.arange(0,iso_store.size,1,dtype=int))
        for n in np.arange(0,clvd_store.size,1,dtype=int)
        for i in np.arange(0,strike_store.size,1,dtype=int)
        for j in np.arange(0,dip_store.size,1,dtype=int)
        for k in np.arange(0,rake_store.size,1,dtype=int)]
    print(parameter)
    outs =pool.map_async(syn_sum_amp_iso_clvd,tqdm(parameter))
    pool.close()
    pool.join()
    output=outs.get()
    data=np.array(output)
    data_reshape=data.reshape(iso_store.size,clvd_store.size,strike_store.size,dip_store.size,rake_store.size)
    index=np.unravel_index(np.argmin(data_reshape, axis=None), data_reshape.shape)
    m_rr,m_tt,m_pp,m_rp,m_rt,m_tp=functions.nmtensor(strike_store[index[2]],dip_store[index[3]],rake_store[index[4]],iso_store[index[0]],clvd_store[index[1]])
    fault=np.array([strike_store[index[2]],dip_store[index[3]],rake_store[index[4]]])
    m_inverse=np.array([m_rr,m_tt,m_pp,m_rt,m_rp,m_tp])
    d_syn=np.dot(Gt,m_inverse)
    return data,m_inverse,d_syn



