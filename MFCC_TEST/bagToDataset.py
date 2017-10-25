#!/usr/bin/python
#
# Copyright (c) 2017, Georgia Tech Research Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#  \author Michael Park (Healthcare Robotics Lab, Georgia Tech.)

# To Use
# Run pubPyAudio.py to read and publish data
# Or Run, rosbag record -O file.bag /hrl_manipulation_task/wrist_audio
# Run this Script to reconstruct the Audio Wav from data collected in rosbag


# Start up ROS pieces.
PKG = 'my_package'
# import roslib; roslib.load_manifest(PKG)
import rosbag
import rospy
import os, copy, sys

## from hrl_msgs.msg import FloatArray
## from std_msgs.msg import Float64
from hrl_anomaly_detection.msg import audio

# util
import numpy as np
import math
import pyaudio
import struct
import array
try:
    from features import mfcc
except:
    from python_speech_features import mfcc
from scipy import signal, fftpack, conj, stats

import scipy.io.wavfile as wav

def dataset_creator(): 
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 3 #needs to be dynamic, length of audio data should be equal

    bag_files = ['data1.bag', 'data2.bag', 'data3.bag', 'data4.bag', 'data5.bag']

    static_ar = []
    dynamic_ar = []
    audio_store = []
    audio_samples = []
    time_store = []
    for bagfile in bag_files:
        for topic, msg, t in rosbag.Bag('./bagfiles/'+bagfile).read_messages():
            #print msg
            if msg._type == 'hrl_anomaly_detection/audio':
                audio_store.append(np.array(msg.audio_data, dtype=np.int16))
            elif msg._type == 'visualization_msgs/Marker':
                if msg.id == 0: #id 0 = static
                    static_ar.append(np.array(msg.pose.position))#, dtype=np.float64))
                elif msg.id == 9: #id 9 = dynamic
                    dynamic_ar.append(np.array(msg.pose.position))#, dtype=np.float64))
            time_store.append(t)
        audio_samples.append(audio_store)



    # if(cnt):
    #     static = static[:-1] #static or dynamic whichever was collected first, look at AR tag number
    #     audio = audio[:-1]

    # relative = static-dynamic

    ##**** Audio Processing ****##
    #copy the frame and insert to lengthen
    data_store_long = []
    baglen = len(audio_store)
    num_frames = RATE/CHUNK * RECORD_SECONDS
    recovered_len = num_frames/baglen

    for frame in audio_store:
        for i in range(0, recovered_len): ##This happens to work cuz recovered len is 2 and num of channels is 2 ???
            data_store_long.append(frame)

    print data_store_long
    numpydata = np.hstack(data_store_long)
    numpydata = np.reshape(numpydata, (len(numpydata)/CHANNELS, CHANNELS))    

    wav.write('test.wav', RATE, numpydata)

    # for audio_file in audio_filename:
    #     y, sr = librosa.load(audio_file, mono=True)
    #     print y.shape

    #     #first divide into time chunks then convert to mfcc
    #     #Without Converting to Binary
    #     mfccs = librosa.feature.mfcc(y, n_mfcc=2) #default hop_length=512
    #     # print mfccs
    #     # print mfccs.shape
        
    #     # normalize - for feeding into LSTM
    #     min_mfcc = np.min(mfccs)
    #     max_mfcc = np.max(mfccs)
    #     mfccs = (mfccs - min_mfcc) / (max_mfcc - min_mfcc)
    #     print mfccs.dtype, min_mfcc, max_mfcc

    #     print mfccs
    #     print mfccs.shape

    ##**** Image Processing ****##


def main():
    rospy.init_node(PKG)
    dataset_creator()

if __name__ == '__main__':
    main()


