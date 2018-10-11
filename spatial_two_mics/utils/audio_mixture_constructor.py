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
                 force_delays=None,
                 normalize_audio_by_std=True,
                 mixture_duration=2.0,
                 precision=0.01,
                 freqs_included=7):
        """
        :param fs: sampling rate
        :param n_fft: FFT window size
        :param win_len: The window will be of length win_length and
        then padded with zeros to match n_fft.
        If unspecified, defaults to win_length = n_fft.
        :param hop_len: number audio of frames between STFT columns.
        If unspecified, defaults win_length / 4.
        :param force_delays: list of delays to be forced in the
        source signals -1 or 1 integer delay for the microphones
        mixtures, if is [0, 0] then no delay would be forced
        :param normalize_audio_by_std: if the loaded wavs would be
        normalized by their std values
        :param mixture_duration: the duration on which the mixture
        would be created (in seconds)
        :param precision: The precision as a floating number e.g. 0.01
        if you are using floating point delays between your source
        signals for each mixture
        :param freqs_included: How many frequencies should be
        included in the sinc function before convolving it with the
        true signal in order to upsample it (1/precision) times more
        and shift it in order to get the truly delayed signal.
        """
        self.mixture_duration = mixture_duration
        self.n_fft = n_fft
        self.win_len = win_len
        self.hop_len = hop_len
        self.normalize_audio_by_std = normalize_audio_by_std
        self.force_delays = force_delays
        self.precision = precision
        self.freqs_included = freqs_included

        xs = np.linspace(-self.freqs_included,
                         self.freqs_included,
                         2. * self.freqs_included / self.precision)
        self.windowed_sinc = np.sinc(xs)

    @staticmethod
    def load_wav(source_info):
        return wavfile.read(source_info['wav_path'])

    def get_stft(self,
                 signal):

        return stft(signal,
                    n_fft=self.n_fft,
                    win_length=self.win_len,
                    hop_length=self.hop_len)

    def force_delay_on_signal(self,
                              signal,
                              delay):
        if delay >= 0:
            return signal[delay:]
        else:
            return signal[:delay]

    def enforce_float_delays(self,
                             source_signals,
                             delays_for_sources,
                             fs):
        """!
        For 2 microphone enforce a floating point number delay with some
        selected precision and apply that for all sources that would
        be given. Also make sure that the required to be returned
        wavs have to have a length equal to the duration"""
        upsampling_rate = int(1. / self.precision)
        duration_in_samples = int(self.mixture_duration * fs) - 1
        decimals = int(np.log10(upsampling_rate))
        n_augmentation_zeros = upsampling_rate - 1

        rounded_taus = np.around(delays_for_sources, decimals=decimals)
        taus_samples = upsampling_rate * rounded_taus
        taus_samples = taus_samples.astype(int)

        mic_signals = {'m1':[], 'm2':[]}
        for src_id, source_sig in enumerate(source_signals):
            sig_len = source_sig.shape[0]
            augmented_signal = np.zeros(
                sig_len + (sig_len - 1) * n_augmentation_zeros)
            augmented_signal[::upsampling_rate] = source_sig
            est_augmented_sig = np.convolve(augmented_signal,
                                            self.windowed_sinc,
                                            mode='valid')

            tau_in_samples = taus_samples[src_id]
            if tau_in_samples > 0:
                source_in_mic1 = est_augmented_sig[
                                 tau_in_samples:][::upsampling_rate]
                source_in_mic2 = est_augmented_sig[
                                 :-tau_in_samples][::upsampling_rate]
            elif tau_in_samples < 0:
                source_in_mic1 = est_augmented_sig[
                                 :tau_in_samples][::upsampling_rate]
                source_in_mic2 = est_augmented_sig[
                                 -tau_in_samples:][::upsampling_rate]
            else:
                source_in_mic1 = est_augmented_sig[::upsampling_rate]
                source_in_mic2 = est_augmented_sig[::upsampling_rate]

            # check the duration which is very important
            if (len(source_in_mic1) < duration_in_samples or
                    len(source_in_mic2) < duration_in_samples):
                raise ValueError("Duration given: {} could "
                                 "not be sufficed before the gven source"
                                 " signal has a lesser duration of {} "
                                 "after the float delay.".format(
                    duration_in_samples, len(source_in_mic1)))

            mic_signals['m1'].append(
                        source_in_mic1[:duration_in_samples])
            mic_signals['m2'].append(
                        source_in_mic2[:duration_in_samples])

        return mic_signals

    def construct_mic_signals(self,
                              source_signals,
                              delays_for_sources):
        """!
        This function might extend to any real delay by interpolation
        of the source signals or just forcing a delay over the sources.
        After that it returns a dictionary containing a list of signals
        for each microphone. also cropped to the duration specified.

        :return mic_signals ={ 'm1': [s1, s2, ..., sn], 'm2': same }
        """

        fs = source_signals[0][1]
        assert all([sr == fs for (s, sr) in source_signals]), 'When ' \
               'trying to enforce the delays over the source signals ' \
               'the fs should be the same for all sources!'

        if self.force_delays is None:
            mic_signals = self.enforce_float_delays(
                           [s for (s, sr) in source_signals],
                           delays_for_sources,
                           fs)

        else:
        # naive way in order to force a delay for DUET algorithm
            m1_delays = self.force_delays
            m2_delays = self.force_delays[::-1]

            cropped_signals = [s[:int(self.mixture_duration * fs)]
                               for (s, sr) in source_signals]

            mic_signals = {
                'm1': [self.force_delay_on_signal(s, m1_delays[i])
                       for (i, s) in enumerate(cropped_signals)],
                'm2': [self.force_delay_on_signal(s, m2_delays[i])
                       for (i, s) in enumerate(cropped_signals)]
            }

        return mic_signals

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
        n_sources = len(source_signals)

        mic_signals = self.construct_mic_signals(source_signals,
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
            'sources_raw': mic_signals['m1'],
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
                                              force_delays=[-1, 1])

    mixture_info = me.mixture_info_example()

    import spatial_two_mics.data_generator.source_position_generator \
        as  position_generator

    # add some randomness in the generation of the positions
    random_positioner = position_generator.RandomCirclePositioner()
    positions_info = random_positioner.get_sources_locations(2)
    mixture_info['positions'] = positions_info

    tf_mixtures = mixture_creator.construct_mixture(mixture_info)

    pprint(tf_mixtures)

if __name__ == "__main__":
    example_of_usage()