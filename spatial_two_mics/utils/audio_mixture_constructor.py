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
                 force_all_signals_one_sample_delay=False,
                 normalize_audio_by_std=True,
                 mixture_duration=2.0,
                 force_no_delay=False):
        """
        :param fs: sampling rate
        :param n_fft: FFT window size
        :param win_len: The window will be of length win_length and
        then padded with zeros to match n_fft.
        If unspecified, defaults to win_length = n_fft.
        :param hop_len: number audio of frames between STFT columns.
        If unspecified, defaults win_length / 4.
        :param force_all_signals_one_sample_delay: if true then forces a
        -1 or 1 integer delay for the microphones mixtures
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
        self.force_all_signals_one_sample_delay =  \
            force_all_signals_one_sample_delay
        self.force_no_delay = force_no_delay

    @staticmethod
    def load_wav(source_info):
        return wavfile.read(source_info['wav_path'])

    def get_stft(self,
                 signal):

        return stft(signal,
                    n_fft=self.n_fft,
                    win_length=self.win_len,
                    hop_length=self.hop_len)

    def construct_delayed_signals(self,
                                  signals,
                                  taus):
        """!
        This function might extend to any real delay by interpolation
        of the source signals

        :return mic_signals ={ 'm1': [s1, s2, ..., sn], 'm2': same }
        """

        if self.force_no_delay:
            return {'m1': signals, 'm2':signals}

        # naive way in order to force a delay for DUET algorithm
        if self.force_all_signals_one_sample_delay:
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
                               mixture_info):
        """!
        This function constructs the mixture for each mic (m1,
        m2) in the following way:
        m1(t) = a1*s1(t) + ... + an*sn(t)
        m2(t) = a1*s1(t+d1) + ... + an*sn(t+dn)

        by also cutting them off to self.min_samples

        :return
        mixture_info = {
            'm1_raw': numpy array containing the raw m1 signal,
            'm2_raw': numpy array containing the raw m2 signal,
            'm1_tf': numpy array containing the m1 TF representation,
            'm2_tf': numpy array containing the m2 TF representation,
            'sources_raw': a list of numpy 1d vectors containing the
            sources ,
            'sources_tf': a list of numpy 2d vectors containing the
             TF represeantations of the sources ,
            'amplitudes': the weights that each source contributes to
            the mixture of the second microphone
        }
        """
        positions = mixture_info['positions']
        source_signals = [(s['wav'], s['fs'])
                          for s in mixture_info['sources_ids']]

        cropped_signals = [s[:int(self.mixture_duration * fs)]
                           for (s, fs) in source_signals]
        n_sources = len(cropped_signals)

        self.force_no_delay = True
        mic_signals = self.construct_delayed_signals(cropped_signals,
                                                     positions['taus'])

        m1 = sum([positions['amplitudes'][i] * mic_signals['m1'][i]
                  for i in np.arange(n_sources)])

        m2 = sum([positions['amplitudes'][i] * mic_signals['m2'][i]
                  for i in np.arange(n_sources)])

        sources_spectra = [self.get_stft(s) for s in mic_signals['m1']]

        m1_tf = self.get_stft(m1)
        m2_tf = self.get_stft(m2)

        mixture_info = {
            'm1_raw': m1,
            'm2_raw': m2,
            'm1_tf': m1_tf,
            'm2_tf': m2_tf,
            'sources_raw': cropped_signals,
            'sources_tf': sources_spectra,
            'amplitudes': positions['amplitudes']
        }

        return mixture_info

    def construct_mixture(self,
                          mixture_info):
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


        :return tf_representations = {
            'm1_raw': numpy array containing the raw m1 signal,
            'm2_raw': numpy array containing the raw m2 signal,
            'm1_tf': numpy array containing the m1 TF representation,
            'm2_tf': numpy array containing the m2 TF representation,
            'sources_raw': a list of numpy 1d vectors containing the
            sources ,
            'sources_tf': a list of numpy 2d vectors containing the
             TF represeantations of the sources ,
            'amplitudes': the weights that each source contributes to
            the mixture of the second microphone
        }
        """

        for i, source_info in enumerate(mixture_info['sources_ids']):
            fs, wav = self.load_wav(source_info)
            if self.normalize_audio_by_std:
                wav = wav / np.std(wav)
            mixture_info['sources_ids'][i]['fs'] = int(fs)
            mixture_info['sources_ids'][i]['wav'] = wav

        tf_representations = self.get_tf_representations(mixture_info)

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
                                              mixture_duration=2.0,
                                              force_all_signals_one_sample_delay=True)

    mixture_info = me.mixture_info_example()

    tf_mixtures = mixture_creator.construct_mixture(mixture_info)

    pprint(tf_mixtures)

if __name__ == "__main__":
    example_of_usage()