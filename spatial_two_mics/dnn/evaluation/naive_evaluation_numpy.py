"""!
@brief A naive implementation of how we evaluate the masks that are
derived --> reconstruct the source signals and also extract SDR,
SIR and SBR for the reconstructed signals and the true signals

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import numpy as np
import librosa


def bss_eval(sep, i, sources):
    # Current target
    min_len = min([len(sep), len(sources[i])])
    sources = sources[:, :min_len]
    sep = sep[:min_len]
    target = sources[i]

    # Target contribution
    s_target = target * np.dot(target, sep.T) / np.dot(target, target.T)

    # Interference contribution
    pse = np.dot(np.dot( sources, sep.T),
    np.linalg.inv(np.dot( sources, sources.T))).T.dot( sources)
    e_interf = pse - s_target

    # Artifact contribution
    e_artif = sep - pse

    # Interference + artifacts contribution
    e_total = e_interf + e_artif

    # Computation of the log energy ratios
    sdr = 10*np.log10(sum(s_target**2) / sum(e_total**2));
    sir = 10*np.log10(sum(s_target**2) / sum(e_interf**2));
    sar = 10*np.log10(sum((s_target + e_interf)**2) / sum(e_artif**2));

    # Done!
    return sdr, sir, sar


def naive_cpu_bss_eval(embedding_labels,
                       mix_real_tf,
                       mix_imag_tf,
                       sources_raw,
                       n_sources,
                       batch_index=0):

    mix_stft = mix_real_tf + 1j*mix_imag_tf

    if mix_stft.shape == embedding_labels.shape:
        embedding_clustered = embedding_labels
    else:
        embedding_clustered = embedding_labels.reshape(
                              mix_stft.shape[::-1]).T

    sdr_t, sir_t, sar_t = 0., 0., 0.
    for i in np.arange(n_sources):
        embed_mask = mix_stft*(embedding_clustered == i)
        reconstructed = librosa.core.istft(embed_mask,
                                           hop_length=128,
                                           win_length=512)
        bss_results = [bss_eval(reconstructed, j, sources_raw)
                       for j in np.arange(n_sources)]

        sdr, sir, sar = sorted(bss_results, key=lambda x: x[0])[-1]
        sdr_t += sdr
        sir_t += sir
        sar_t += sar

        # save_p = '/home/thymios/wavs/'
        # wav_p = os.path.join(save_p,
        #                      'batch_{}_source_{}'.format(
        #                          batch_index + 1, i + 1))
        # librosa.output.write_wav(wav_p, reconstructed, 16000)

    return sdr_t/n_sources, sir_t/n_sources, sar_t/n_sources
