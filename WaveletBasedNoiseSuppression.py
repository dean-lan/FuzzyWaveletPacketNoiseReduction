import pywt

import numpy as np

import matplotlib.pyplot as plt
import torch.version

from Cluster.CFuzzyClustering import CFuzzyCluster
from ImprovedWaveletPacketDenoise.DecomposeLevelSelector import DecomposeLevelSelectorFactory
from ImprovedWaveletPacketDenoise.Decomposer import DecomposerFactory
from ImprovedWaveletPacketDenoise.DenoiseHelper import DenoiseHelper, DenoiseMetricsEntropy
from ImprovedWaveletPacketDenoise.NoiseEstimator import NoiseEstimatorFactory
from ImprovedWaveletPacketDenoise.ThresholdValue import ThresholdFactory
from ImprovedWaveletPacketDenoise.Thresholding import ThresholdingFactory
from MSPicker import DataPreparation
from Utils import SignalGenerator


class WaveletBasedNoiseSuppression(object):
    def __init__(self):
        super(WaveletBasedNoiseSuppression, self).__init__()

    def denoise(self, data, **kwargs):
        length = len(data)

        wavelet = kwargs.get('wavelet', 'db10')
        level = kwargs.get('level', None)

        decomposer_type = kwargs.get('decomposer_type', 'wavelet')
        decompose_level_selector_type = kwargs.get('decompose_level_selector_type', 'preset')
        noise_estimator_type = kwargs.get('noise_estimator_type', 'normal')
        thresholding_type = kwargs.get('thresholding_type', 'garrote')
        threshold_type = kwargs.get('threshold_type', 'global')

        decomposer = DecomposerFactory().create_decomposer(composer_type=decomposer_type)
        level_selector = DecomposeLevelSelectorFactory(). \
            create_selector(selector_type=decompose_level_selector_type)
        noise_estimator = NoiseEstimatorFactory().create_estimator(estimator_type=noise_estimator_type)
        threshold = ThresholdFactory().create_threshold(threshold_type=threshold_type)
        thresholding = ThresholdingFactory().create_thresholding(thresholding_type=thresholding_type)

        level = level_selector.get(data, wavelet, level=level)

        kwargs['level'] = level
        kwargs['wavelet'] = wavelet

        decomposer.decompose(data, **kwargs)

        decomposer.do_save(suffix='before', **kwargs)
        for index in range(len(decomposer)):
            noise_delta = noise_estimator.get(decomposer, index=index, **kwargs)
            d = decomposer[index]
            thd = threshold.get(noise_delta, length, **kwargs)
            d = thresholding.apply(d, thd, index=index, decompositions=decomposer, **kwargs)
            decomposer[index] = d
        decomposer.do_save(suffix='after', **kwargs)
        processed_data = decomposer.recovery()
        return processed_data


if __name__ == '__main__':
    fm = 150
    fs = 1000
    num = 1
    length = 500

    snr = -3.0

    sig = SignalGenerator.ricker(fm, length, fs)
    # sig = SignalGenerator.sinusoidal(1.0, 0.0, fm, 0.5, fs)
    noise = DenoiseHelper('Test').load_noise(num=num, length=length, is_wgn='wgn')

    cf = CFuzzyCluster()
    features = DataPreparation(sig).get_features(wsize=13, fnames=['K', 'P', 'Δ'])
    result = cf.fit(samples=features, n_cluster=2, fuzziness=2.0)
    coeffs = result.get_coefficients(normalized=True, min_in_binary=True)
    print(np.sum(coeffs))

    sns = SignalGenerator.signal_mixture(sig, noise, snr=snr)

    # emd_decomposer = DecomposerFactory().create_decomposer('ceemdan')
    # emd_decomposer.decompose(data=sns[0])
    # pcs = np.asarray([np.corrcoef(sns[0], _d)[0][1] for _d in emd_decomposer])
    # pcs = np.abs(pcs)
    # all_indexs = np.linspace(0, len(emd_decomposer), num=len(emd_decomposer), endpoint=False, dtype=int)
    # minor_components_indexs = np.nonzero(pcs < 0.2)[0]
    # major_components_indexs = np.delete(all_indexs, minor_components_indexs)
    # print('Major: ', major_components_indexs)
    # print('Minor: ', minor_components_indexs)

    cf = CFuzzyCluster()
    features = DataPreparation(sns[0]).get_features(wsize=13, fnames=['K', 'P', 'Δ'])
    result = cf.fit(samples=features, n_cluster=2, fuzziness=2.0)
    coeffs = result.get_coefficients(normalized=True, min_in_binary=True)

    # for _index in minor_components_indexs:
    #     break
    #     emd_decomposer[_index] = np.zeros(len(sns[0]))
    #
    # for _index in major_components_indexs:
    #     d = emd_decomposer[_index]
    #
    #     processed_d = WaveletBasedNoiseSuppression().denoise(data=d, wavelet='db8', level=3,
    #                                                          decomposer_type='WaveletPacket',
    #                                                          decompose_level_selector_type='preset',
    #                                                          noise_estimator_type='node_dependent',
    #                                                          thresholding_type='hard',
    #                                                          threshold_type='level_dependent',
    #                                                          save_to='./Test',
    #                                                          fuzziness=2.0)
    #     emd_decomposer[_index] = processed_d
    #
    # data = emd_decomposer.recovery()
    # print(np.corrcoef(sig, data)[0][1], np.std(sig - data),
    #       DenoiseMetricsEntropy().get(sns[0] - data, entropy_type='sample_entropy'))
    # data = data * coeffs
    # print(np.corrcoef(sig, data)[0][1], np.std(sig - data),
    #      DenoiseMetricsEntropy().get(sns[0] - data, entropy_type='sample_entropy'))
    # print()

    for tt in ['hard', 'soft', 'garrote', 'class']:

        data = WaveletBasedNoiseSuppression().denoise(data=sns[0], wavelet='db8', level=1,
                                                      decomposer_type='WaveletPacket',
                                                      decompose_level_selector_type='preset',
                                                      noise_estimator_type='global',
                                                      threshold_type='level_dependent',
                                                      thresholding_type=tt,
                                                      save_to='./Test',
                                                      fuzziness=2.0)

        print(np.corrcoef(sig, data)[0][1], np.std(sig - data),
              DenoiseMetricsEntropy().get(sns[0] - data, entropy_type='sample_entropy'))

        plt.figure()
        plt.title(tt)
        plt.subplot(411)
        plt.plot(sig)
        plt.subplot(412)
        plt.plot(sns[0])
        plt.subplot(413)
        plt.plot(data)
        plt.subplot(414)
        plt.plot(sns[0] - data)

        cf = CFuzzyCluster()
        features = DataPreparation(data).get_features(wsize=13, fnames=['K', 'P', 'Δ'])
        result = cf.fit(samples=features, n_cluster=2, fuzziness=2.0)
        coeffs = result.get_coefficients(normalized=True, min_in_binary=True)

        # coeffs = np.where(coeffs < 0.25, 0.0, 1.0)

        data = data * coeffs
        print(np.corrcoef(sig, data)[0][1], np.std(sig - data),
              DenoiseMetricsEntropy().get(sns[0] - data, entropy_type='sample_entropy'))

        plt.figure()
        plt.subplot(211)
        plt.plot(data)
        plt.subplot(212)
        plt.plot(sns[0] - data)
        print()

    plt.show()
