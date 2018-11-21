######## using the asdf and obspy to deresponse the pyasdf and save as preprocessed tag in an asdf file
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader
from obspy.clients.seedlink.easyseedlink import create_client
from obspy import read_events,read,read_inventory
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import os
import sys
import datetime
import obspy
import glob
import pyasdf
import obspy
import numpy as np
#import find
from obspy.io.stationxml.core import validate_stationxml
from obspy.geodetics.base import gps2dist_azimuth
     #obspy.core.util.geodetics.base
from pyasdf import ASDFDataSet
with open("error.log","w") as myfile:
     files = glob.glob('*.h5')
     for f in files:
        ds = pyasdf.ASDFDataSet(f)
        event = ds.events[0]
########find some parameters defined in asdf file.
        origin = event.preferred_origin() or event.origins[0]
        event_latitude = origin.latitude
        event_longitude = origin.longitude
        starttime=origin.time
        Statlst=ds.waveforms.list()
        pre_filt = (0.1, 1, 10, 100)
        judge=ds.validate()
        print(judge)
        for _i,staid in enumerate(Statlst):
                name=Statlst[_i]
                st=ds.waveforms[name].raw_recording
                ts=st
               #print(ts)
               #print(name)
                list=ds.waveforms[name].list()
                print(list)
                ##check whether some stations don't contain the station xml
                if 'StationXML' in list: 
                         print('ok')                            
                         inv=ds.waveforms[name].StationXML      
                         print(inv[0]) 
                         ts.detrend("linear")
                         ts.detrend("demean")
                         ts.taper(max_percentage=0.05, type="hann")
                         ts.attach_response(inv)
                         ts.remove_response(output="DISP", pre_filt=pre_filt, zero_mean=False,
                                     taper=False)
                         ts.detrend("linear")
                         ts.detrend("demean")
                         ts.taper(max_percentage=0.05, type="hann")
                         ds.add_waveforms(ts, tag="preprocess", event_id=event)
               ## and reported to the errorlog
                else:
                         print('take care\n'+name+list[0])
                         print(list)
                         #with open("event_station2010_sdyregion.list","w") as myfile:
                         myfile.write("%s\n" %(judge) )
                         myfile.write("%s doesn't contain the stationxml files\n" %(name) )
  



