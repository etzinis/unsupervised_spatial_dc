"""!
@brief This utility serves as a level of abstraction in order to
construct audio mixtures


@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana Champaign
"""

from librosa.core import stft
from pprint import pprint
import numpy as np
import scipy.io.wavfile as wavfile


class AudioMixtureConstructor(object):
    def __init__(self,
                 n_fft=1024,
                 win_len=None,
                 hop_len=None,
                 force_all_signals_integer_delay=False,
                 normalize_audio_by_std=True,
                 mixture_duration=2.0):
        """
        :param fs: sampling rate
        :param n_fft: FFT window size
        :param win_len: The window will be of length win_length and
        then padded with zeros to match n_fft.
        If unspecified, defaults to win_length = n_fft.
        :param hop_len: number audio of frames between STFT columns.
        If unspecified, defaults win_length / 4.
        :param force_all_signals_integer_delay: if true then forces a
        -1, 0 , 1 integer delay for the microphones mixtures
        :param normalize_audio_by_std: if the loaded wavs would be
        normalized by their std values
        :param mixture_duration: the duration on which the mixture
        would be created (in seconds)
        """
        self.mixture_duration = mixture_duration
        self.n_fft = n_fft
        self.win_len = win_len
        self.hop_len = hop_len
        self.normalize_audio_by_std = normalize_audio_by_std

    @staticmethod
    def load_wav(source_info):
        return wavfile.read(source_info['wav_path'])

    def get_stft(self,
                 signal):

        return stft(signal,
                    n_fft=self.n_fft,
                    win_length=self.win_len,
                    hop_length=self.hop_len)

    @staticmethod
    def construct_delayed_signals(signals,
                                  taus,
                                  force_all_signals_delay=False):
        """!
        This function might extend to any real delay by interpolation
        of the source signals
        """

        # naive way in order to force a delay for DUET algorithm
        if force_all_signals_delay:
            delays = []
            if force_all_signals_delay:
                delays = []
                for tau in taus:
                    if tau >= 0:
                        delays.append(1)
                    else:
                        delays.append(-1)
        else:
            raise NotImplementedError("A real value delay should be "
                                      "implemented by utilizing "
                                      "interpolation. Currently "
                                      "Unavailable.")

        delayed_signals = []
        for i, delay in enumerate(delays):
            new_signal = np.roll(signals[i], -delay)
            if delay > 0:
                new_signal[-delay:] = 0.
            elif delay < 0:
                new_signal[:-delay] = 0.
            delayed_signals.append(new_signal)

        return delayed_signals

    def get_tf_representations(self,
                               mixture_info,
                               force_all_signals_delay=False):
        """!
        This function constructs the mixture for each mic (m1,
        m2) in the following way:
        m1(t) = a1*s1(t) + ... + an*sn(t)
        m2(t) = a1*s1(t+d1) + ... + an*sn(t+dn)

        by also cutting them off to self.min_samples
        """
        positions = mixture_info['positions']
        source_signals = [(s['wav'], s['fs'])
                          for s in mixture_info['sources_ids']]

        cropped_signals = [s[:int(self.mixture_duration * fs)]
                           for (s, fs) in source_signals]
        delayed_signals = self.construct_delayed_signals(
            cropped_signals,
            positions['taus'],
            force_all_signals_delay=force_all_signals_delay)

        m1 = sum([positions['amplitudes'][i] * cropped_signals[i]
                  for i in np.arange(len(cropped_signals))])

        m2 = sum([positions['amplitudes'][i] * delayed_signals[i]
                  for i in np.arange(len(delayed_signals))])

        sources_spectra = [self.get_stft(s) for s in cropped_signals]

        delayed_sources_spectra = [self.get_stft(s)
                                   for s in delayed_signals]

        m1_tf = stft(m1, n_fft=1024, win_length=320)
        m2_tf = stft(m2, n_fft=1024, win_length=320)

        mixture_info = {
            'm1_raw': m1,
            'm2_raw': m2,
            'm1_tf': m1_tf,
            'm2_tf': m2_tf,
            'sources_raw': cropped_signals,
            'sources_tf': sources_spectra,
            'delayed_sources_raw': delayed_signals,
            'delayed_sources_tf': delayed_sources_spectra,
            'amplitudes': positions['amplitudes']
        }

        return mixture_info

    def construct_mixture(self,
                          mixture_info,
                          force_all_signals_delay=False):
        """! The whole processing for getting the mixture signals for
        the two mics and the positions is done here.

        :param mixture_info
        {
            'positions': example
            'sources_ids':
            [       {
                        'gender': combination_info.gender
                        'sentence_id': combination_info.sentence_id
                        'speaker_id': combination_info.speaker_id
                        'wav_path': the wav_path for the file
                    } ... ]
        }

        :param force_all_signals_delay whether you need to force an
        integer delay of -1, 0, 1

        :return
        """
        n_sources = len(mixture_info['sources_ids'])

        for i, source_info in enumerate(mixture_info['sources_ids']):
            fs, wav = self.load_wav(source_info)
            if self.normalize_audio_by_std:
                wav = wav / np.std(wav)
            mixture_info['sources_ids'][i]['fs'] = int(fs)
            mixture_info['sources_ids'][i]['wav'] = wav

        tf_representations = self.get_tf_representations(
            mixture_info,
            force_all_signals_delay=force_all_signals_delay
        )

        return tf_representations


def example_of_usage():
    """!
    How the class of Audio mixtures should be called"""

    import os
    import sys
    root_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../../')
    sys.path.insert(0, root_dir)
    import spatial_two_mics.examples.mixture_example as me

    mixture_creator = AudioMixtureConstructor(n_fft=1024,
                                              win_len=1024,
                                              hop_len=512,
                                              mixture_duration=2.0)

    example_mixture_info = me.mixture_info_example()

    tf_mixtures = mixture_creator.construct_mixture(
                                  example_mixture_info,
                                  force_all_signals_delay=True)

    pprint(tf_mixtures)

if __name__ == "__main__":
    example_of_usage()