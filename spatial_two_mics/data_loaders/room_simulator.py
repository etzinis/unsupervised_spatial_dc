"""!
@brief A room simulation class for simple shoebox rooms. 
Mics are randomly places on the unit circle in the center of the room.
Sources are either placed at specified locations or radomly anywhere in the room.

@author Jonah Casebeer {jonahmc2@illinois.edu}
@copyright University of Illinois at Urbana Champaign
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import pyroomacoustics as pya



class room_sim():
    def __init__(self, w = 10, h = 10, fs = 16000, max_order = 0):
        """
        w: int, width of shoebox room
        h: int, heigh of showbox room
        fs: int, sampling rate of the room simulator
        max_order: int, number of echoes to estimate via image source method. 0 corresponds to anechoic
        """
        self.w = w
        self.h = h
        self.fs = fs
        self.max_order = max_order
        
    def make_room(self, num_mics, num_source, source_data, source_locs = 'random'):
        """
        num_mics: int, number of mics in room to simulate
        num_source: int, number of sources in room to simulate
        source_date: a matrix of size num_sources x num_samples
        """
        room  = pya.ShoeBox([self.w, self.h], fs = self.fs, max_order = self.max_order)

        for source in range(num_source):
            if source_locs == 'random':
                source_loc = np.array([torch.rand(1).numpy()[0] * self.w, torch.rand(1).numpy()[0] * self.h])
            else:
                source_loc = source_locs[source]
            room.add_source(source_loc, signal = source_data[source])

        theta = torch.rand((num_mics,1)).numpy() * 2 * np.pi
        mic_locs = np.array([self.w/2, self.h/2]) + np.hstack(( np.cos(theta), np.sin(theta )))
        mic_arr = pya.MicrophoneArray(mic_locs, room.fs)
        room.add_microphone_array(mic_arr)
        room.image_source_model()
        return room, mic_arr
        
    def simulate(self, num_mics, num_source, source_data, source_locs = 'random'):
        """
        num_mics: int, number of mics in room to simulate
        num_source: int, number of sources in room to simulate
        source_date: a matrix of size num_sources x num_samples
        source_locs: DEFAULT, STRING for random locations. Can take a matrix of size num_sources x 2
        """
        room, mic_arr = self.make_room(num_mics, num_source, source_data, source_locs)
        room.simulate()
        room.plot(img_order=0)
        plt.show()
        return mic_arr.signals
    
    def simulate_batch(self, num_mics, num_source, source_data, source_locs = 'random'):
        """
        This methods differs from a simple for loop around batch in that it zero pads the signals to fit in a matrix.
        num_mics: int, number of mics in room to simulate
        num_source: int, number of sources in room to simulate
        source_date: a matrix of size num_batches x num_sources x num_samples
        source_locs: DEFAULT, STRING for random locations. Can take a matrix of size num_sources x 2
        """
        result = []
        longest = -np.inf
        for bt in range(source_data.shape[0]):
            room, mic_arr = self.make_room(num_mics, num_source, source_data[bt], source_locs)
            room.simulate()
            room.plot(img_order=0)
            plt.show()
            sig = mic_arr.signals
            if sig.shape[1] > longest:
                longest = sig.shape[1]
            result.append( sig )
            
        
        final = np.zeros((source_data.shape[0], num_source, longest))
        for bt in range(final.shape[0]):
            sigs = result[bt]
            final[bt,:,:sigs.shape[1]] = sigs
        return final
        