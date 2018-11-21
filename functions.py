#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this code is intended to calculate the different kind of misfits and gengerate the moment tensors from DC and from DC plus clvd






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
from obspy.signal.tf_misfit import tfem,tfpm
import numba
from multiprocessing import Pool
from time import sleep
import tqdm





def synth_generator3(uz_mtt,uz_mpp,uz_mrp,uz_mrt,uz_mtp,uz_mrr,mtt,mpp,mrp,mrt,mtp,mrr):
   uz=uz_mtt*mtt+uz_mpp*mpp+uz_mrp*mrp+uz_mrt*mrt+uz_mtp*mtp+uz_mrr*mrr
   return uz


def synth_generator4(ue_mtt,ue_mpp,ue_mrp,ue_mrt,ue_mtp,ue_mrr,mtt,mpp,mrp,mrt,mtp,mrr):
   ue=ue_mtt*mtt+ue_mpp*mpp+ue_mrp*mrp+ue_mrt*mrt+ue_mtp*mtp+ue_mrr*mrr
   return ue

def synth_generator5(un_mtt,un_mpp,un_mrp,un_mrt,un_mtp,un_mrr,mtt,mpp,mrp,mrt,mtp,mrr):
   un=un_mtt*mtt+un_mpp*mpp+un_mrp*mrp+un_mrt*mrt+un_mtp*mtp+un_mrr*mrr
   return un






def DC_MT(strike,dip,rake):
    phi = np.deg2rad(strike)
    delta1 = np.deg2rad(dip)
    lambd = np.deg2rad(rake)
    #az= np.deg2rad(az)
    m_tt = (- np.sin(delta1) * np.cos(lambd) * np.sin(2. * phi) -np.sin(2. * delta1) * np.sin(phi)**2. * np.sin(lambd)) 
    m_pp = (np.sin(delta1) * np.cos(lambd) * np.sin(2. * phi) - np.sin(2. * delta1) * np.cos(phi)**2. * np.sin(lambd)) 
    m_rr = (np.sin(2. * delta1) * np.sin(lambd)) 
    m_rp = (- np.cos(phi) * np.sin(lambd) * np.cos(2. * delta1) + np.cos(delta1) * np.cos(lambd) * np.sin(phi)) 
    m_rt = (- np.sin(lambd) * np.sin(phi) * np.cos(2. * delta1) - np.cos(delta1) * np.cos(lambd) * np.cos(phi)) 
    m_tp = (- np.sin(delta1) * np.cos(lambd) * np.cos(2. * phi) - np.sin(2. * delta1) * np.sin(2. * phi) * np.sin(lambd) / 2.)
    return m_rr,m_tt,m_pp,m_rp,m_rt,m_tp



#####this function is used for the moment tensors 
def nmtensor(strike, dip,rake,iso,clvd):
    phi = np.deg2rad(strike)
    delta1 = np.deg2rad(dip)
    lambd = np.deg2rad(rake)
    sstr=np.sin(phi)
    cstr=np.cos(phi)
    sstr2=2*sstr*cstr
    cstr2=1-2*sstr*sstr
    sdip=np.sin(delta1)
    cdip=np.cos(delta1)
    sdip2=2*sdip*cdip
    cdip2=1-2*sdip*sdip
    crak=np.cos(lambd)
    srak=np.sin(lambd)
    n=np.zeros(3)
    v=np.zeros(3)
    N=np.zeros(3)
    ##### equation 2 in (Zhu, 2013)
    dum=np.sqrt(2/3)*iso  
    tensor=np.zeros((3,3))
    #####the isotropic part
    tensor[0][0] = tensor[1][1] = tensor[2][2] = dum
    tensor[0][1] = tensor[0][2] = tensor[1][2] = 0.
    dev = 1.-iso*iso
    if dev>0:
       dev = np.sqrt(dev)
       dum = dev*np.sqrt(1.-clvd*clvd)
       tensor[0][0] += -dum*(sdip*crak*sstr2+sdip2*srak*sstr*sstr);
       tensor[0][1] +=  dum*(sdip*crak*cstr2+0.5*sdip2*srak*sstr2);
       tensor[0][2] += -dum*(cdip*crak*cstr+cdip2*srak*sstr);
       tensor[1][1] +=  dum*(sdip*crak*sstr2-sdip2*srak*cstr*cstr);
       tensor[1][2] +=  dum*(cdip2*srak*cstr-cdip*crak*sstr);
       tensor[2][2] +=  dum*sdip2*srak;
       if clvd>0.0001 and clvd<-0.0001:
          n[0] = -sdip*sstr
          n[1] = sdip*cstr
          n[2] = -cdip
          v[0] = crak*cstr+srak*cdip*sstr  
          v[1] = crak*sstr-srak*cdip*cstr
          v[2] = -srak*sdip
          N[0] = n[1]*v[2]-n[2]*v[1] 
          N[1] = n[2]*v[0]-n[0]*v[2] 
          N[2] = n[0]*v[1]-n[1]*v[0]
          dum = dev*clvd/sqrt(3.);
          tensor[0][0] += dum*(2*N[0]*N[0]-n[0]*n[0]-v[0]*v[0]);
          tensor[0][1] += dum*(2*N[0]*N[1]-n[0]*n[1]-v[0]*v[1]);
          tensor[0][2] += dum*(2*N[0]*N[2]-n[0]*n[2]-v[0]*v[2]);
          tensor[1][1] += dum*(2*N[1]*N[1]-n[1]*n[1]-v[1]*v[1]);
          tensor[1][2] += dum*(2*N[1]*N[2]-n[1]*n[2]-v[1]*v[2]);
          tensor[2][2] += dum*(2*N[2]*N[2]-n[2]*n[2]-v[2]*v[2]);
       tensor[1][0] = tensor[0][1];
       tensor[2][0] = tensor[0][2];
       tensor[2][1] = tensor[1][2];  
    mrr=tensor[2][2]
    mtt=tensor[0][0]
    mpp=tensor[1][1]
    mrp=-tensor[1][2]
    mrt=tensor[0][2]
    mtp=-tensor[0][1]
    return mrr,mtt,mpp,mrp,mrt,mtp






def corr(u1,u2):
    index,value=xcorr(u1,u2,shift_len=0)
    misfit=1.0-value
    return index,misfit



def corr_NEW(u1,u2):
    #leng_shift=0
    leng_shift=0
    CORR=correlate(u1,u2,leng_shift,normalize=True,domain='freq')
    shift,value=xcorr_max(CORR)
    misfit=1.0-value
    return shift,misfit



def corrNEW_amp(u1,u2):
    #leng_shift=0
    #this function take advantage of the amplitude ratio to control the radiation pattern
    M0_fake=np.max(np.abs(u1))/np.max(np.abs(u2))
    M0_fake_abs=np.abs(np.log(M0_fake))/5
    leng_shift=int(1/4*u1.size)
    CORR=correlate(u1,u2,leng_shift,normalize=True,domain='freq')
    norm=np.sum(CORR)
    shift,value=xcorr_max(CORR)
    misfit=1.0-value/norm+2*M0_fake_abs
    return shift,misfit



##############take advantage of the CC and ratio of the amplitude

def cc_amp(u1,u2):
    ### here is no timeshift for the assessment of misfit u1 should be the observed data while the u2 should be synthetics
    M0_fake_max=np.max(u1)/np.max(u2)
    M0_fake_min=np.min(u1)/np.min(u2)
    M0_fake=np.max(np.abs(u1))/np.max(np.abs(u2))
    #M0_fake=np.sqrt(np.dot(u1,u1)/np.dot(u2,u2))
    #e=np.abs(u1-M0_fake*u2)/np.sqrt(np.dot(np.abs(u1),np.abs(M0_fake*u2)))
    #e=np.abs(u1-M0_fake*u2)/np.sqrt((np.abs(u1)*np.abs(M0_fake*u2)))  ##### unstable
    #e_L1=np.mean(np.abs(e))
    #e_L2=(np.sqrt(np.sum((e*e))))/u1.size
    #e_fll=(e_L1+e_L2+np.sqrt(2*e_L1*e_L1+2*e_L2*e_L2))/4
    leng_shift=int(1/4*u1.size)    
    CORR=correlate(u1,u2*M0_fake,leng_shift,normalize=True,demean=True,domain='freq')
    #####keep the 
    shift,value=xcorr_max(CORR,abs_max=False)
    misfit=1-value
    shift_threshold=5/0.07
    if np.abs(shift)<shift_threshold:
       shift_misfit=0
    else:
       shift_misfit=(np.abs(shift)-shift_threshold)/shift_threshold
    #shift_misfit=np.abs(np.log(shift_misfit))
    ###here add the misfit from the errors from the amplitude
    M0_fake_abs=np.abs(np.log(np.abs(M0_fake_max)))/5+np.abs(np.log(np.abs(M0_fake_min)))/5
    e_fll=misfit+M0_fake_abs+shift_misfit
    #print(e,e_L1,e_L2,e_fll)
    return e_fll







def TF_fem(u1,u2):
    fmin=1/80
    fmax=1/30
    dt=0.07
    misfit=tfem(u1,u2,dt,fmin,fmax)
    max_value=np.max(np.abs(misfit))
    return max_value

def TF_fpm(u1,u2):
    fmin=1/80
    fmax=1/30
    dt=0.07
    #nf=100
    misfit=tfpm(u1,u2,dt,fmin,fmax,nf=100)
    max_value=np.max(np.abs(misfit))
    return max_value










######this function does not consider the amplitude change of the three components
def L1_L2misfit(u1,u2):
    ### here is no timeshift for the assessment of misfit u1 should be the observed data while the u2 should be synthetics
    M0_fake=np.max(u1)/np.max(u2)
    e=np.abs(u1-M0_fake*u2)/np.sqrt(np.dot(np.abs(u1),np.abs(M0_fake*u2)))
    e_L1=np.mean(np.abs(e))
    e_L2=(np.sqrt(np.sum((e*e))))/u1.size
    e_fll=(e_L1+e_L2+np.sqrt(2*e_L1*e_L1+2*e_L2*e_L2))/4
    #print(e,e_L1,e_L2,e_fll)
    return e_fll

#######intended to implement the consideration of the change of the amplitude of the three components
####### here should be input the observed data and the synthetics and return a misfit for a 3 components
def L1_L2misfit_station(d1,d2,d3,u1,u2,u3):
    ### here is no timeshift for the assessment of misfit u1 should be the observed data while the u2 should be synthetics
    M1_fake=np.max(d1)/np.max(u1)
z    M2_fake=np.max(d2)/np.max(u2)
    M3_fake=np.max(d3)/np.max(u3)
    M0_fake=(M1_fake+M2_fake+M3_fake)/3
    E1=np.abs(d1-M0_fake*u1)/np.sqrt(np.dot(np.abs(d1),np.abs(M0_fake*u1)))
    E1_L1=np.mean(np.abs(E1))
    E1_L2=(np.sqrt(np.sum((E1*E1))))/d1.size
    E1_fll=(E1_L1+E1_L2+np.sqrt(2*E1_L1*E1_L1+2*E1_L2*E1_L2))/4

    E2=np.abs(d2-M0_fake*u2)/np.sqrt(np.dot(np.abs(d2),np.abs(M0_fake*u2)))
    E2_L1=np.mean(np.abs(E2))
    E2_L2=(np.sqrt(np.sum((E2*E2))))/d2.size
    E2_fll=(E2_L1+E2_L2+np.sqrt(2*E2_L1*E2_L1+2*E2_L2*E2_L2))/4

    E3=np.abs(d3-M0_fake*u3)/np.sqrt(np.dot(np.abs(d3),np.abs(M0_fake*u3)))
    E3_L1=np.mean(np.abs(E3))
    E3_L2=(np.sqrt(np.sum((E3*E3))))/d3.size
    E3_fll=(E3_L1+E3_L2+np.sqrt(2*E3_L1*E3_L1+2*E3_L2*E3_L2))/4
    #print(e,e_L1,e_L2,e_fll)
    E_fll=E1_fll+E2_fll+E3_fll
    return E_fll


######## here intended to control the amplitude difference between the synthetics and the observed waveforms

def L1_L2misfit_amp(u1,u2):
    ### here is no timeshift for the assessment of misfit u1 should be the observed data while the u2 should be synthetics
    M0_fake_max=np.max(u1)/np.max(u2)
    M0_fake_min=np.min(u1)/np.min(u2)
    M0_fake=np.max(np.abs(u1))/np.max(np.abs(u2))
    #M0_fake=np.sqrt(np.dot(u1,u1)/np.dot(u2,u2))
    e=np.abs(u1-M0_fake*u2)/np.sqrt(np.dot(np.abs(u1),np.abs(M0_fake*u2)))
    #e=np.abs(u1-M0_fake*u2)/np.sqrt((np.abs(u1)*np.abs(M0_fake*u2)))  ##### unstable
    e_L1=np.mean(np.abs(e))
    e_L2=(np.sqrt(np.sum((e*e))))/u1.size
    e_fll=(e_L1+e_L2+np.sqrt(2*e_L1*e_L1+2*e_L2*e_L2))/4
    ###here add the misfit from the errors from the amplitude
    M0_fake_abs=np.abs(np.log(np.abs(M0_fake_max)))/5+np.abs(np.log(np.abs(M0_fake_min)))/5
    e_fll=e_fll+M0_fake_abs
    #print(e,e_L1,e_L2,e_fll)
    return e_fll

def L1_L2misfit_amp_deshifteff(u1,u2):
    ### here is no timeshift for the assessment of misfit u1 should be the observed data while the u2 should be synthetics
    M0_fake_max=np.max(u1)/np.max(u2)
    M0_fake_min=np.min(u1)/np.min(u2)
    M0_fake=np.max(np.abs(u1))/np.max(np.abs(u2))
    #M0_fake=np.sqrt(np.dot(u1,u1)/np.dot(u2,u2))
    e=np.abs(u1-M0_fake*u2)/np.sqrt(np.dot(np.abs(u1),np.abs(M0_fake*u2)))
    #e=np.abs(u1-M0_fake*u2)/np.sqrt((np.abs(u1)*np.abs(M0_fake*u2)))  ##### unstable
    e_L1=np.mean(np.abs(e))
    e_L2=(np.sqrt(np.sum((e*e))))/u1.size
    ######here is intended to deliminate the effect from the phase shift which could lead to an artefact of the misfit
    if e_L1+e_L2 >1:
      print('phase shift')
      e_fll=1
    else:
      e_fll=(e_L1+e_L2+np.sqrt(2*e_L1*e_L1+2*e_L2*e_L2))/4
    ###here add the misfit from the errors from the amplitude
    M0_fake_abs=np.abs(np.log(M0_fake_max))/5+np.abs(np.log(M0_fake_min))/5
    e_fll=e_fll+M0_fake_abs
    #print(e,e_L1,e_L2,e_fll)
    return e_fll








    ###this function is returning a misfit only the misfit of the relative amplitude followed lupei Zhu's theory in 2015 Lushan
def L2misfit(u1,u2):
    M0_fake=np.sqrt(np.dot(u1,u1))/np.sqrt(np.dot(u2,u2))
    e=u1-M0_fake*u2
    e_l2=np.sqrt(np.dot(e,e))
    return e_l2


    ###this function is returning a misfit only the misfit of the relative amplitude followed lupei Zhu's theory and incorporate the 
    ### ratio of the amplitudes

def L2misfit_amp(u1,u2):
    M0_fake=np.sqrt(np.dot(u1,u1))/np.sqrt(np.dot(u2,u2))
    e=u1-M0_fake*u2
    e_l2=np.sqrt(np.dot(e,e))
    M0_fake_abs=np.abs(np.log(M0_fake))/10
    e_fll=100*e_l2+M0_fake_abs
    return e_fll

    ###this function is returning a misfit only the misfit of the relative amplitude followed lupei Zhu's theory in 2015 San Jacinto
def L2misfit_abs(u1,u2):
    #M0_fake=np.sqrt(np.dot(u1,u1))/np.sqrt(np.dot(u2,u2))
    e=u1-u2
    e_l2=np.sqrt(np.dot(e,e))
    #M0_fake_abs=np.abs(np.log(M0_fake))/10
    return e_l2










