import pywt

import numpy as np

import matplotlib.pyplot as plt

from ImprovedWaveletPacketDenoise.Decomposer import DecomposerFactory
from ImprovedWaveletPacketDenoise.DenoiseHelper import DenoiseMetricsEntropy, DenoiseMetricsHighOder, DenoiseHelper
from Utils import SignalGenerator


class DecomposeLevelSelectorABC(object):
    def __init__(self):
        super(DecomposeLevelSelectorABC, self).__init__()

    def get_level_score(self, level_now, **kwargs):
        raise NotImplemented

    def get(self, data, wavelet, **kwargs):
        level = kwargs.get('level', None)
        shown = kwargs.get('shown', False)

        wave_inst = pywt.WaveletPacket(data, wavelet)
        maxlevel = wave_inst.maxlevel
        kwargs['wave_inst'] = wave_inst

        level = maxlevel if level is None else level

        levels = np.linspace(1, level, num=level, endpoint=True, dtype=int)
        scores = np.asarray([self.get_level_score(_l, **kwargs) for _l in levels])

        if shown:
            plt.figure()
            plt.plot(levels, scores)

        opt_index = np.argmin(scores)
        opt_level = levels[opt_index]
        return opt_level


class DecomposeLevelSelectorPreset(DecomposeLevelSelectorABC):
    def __init__(self):
        super(DecomposeLevelSelectorPreset, self).__init__()

    def get_level_score(self, level_now, **kwargs):
        return -level_now


class DecomposeLevelSelectorEntropy(DecomposeLevelSelectorABC):
    def __init__(self):
        super(DecomposeLevelSelectorEntropy, self).__init__()

    def get_level_score(self, level_now, **kwargs):
        entropy_type = kwargs.get('entropy_type', 'fuzzy_entropy')
        wavelet_inst = kwargs.get('wave_inst')
        use_kurtosis = kwargs.get('use_kurtosis', True)

        _nodes = wavelet_inst.get_level(level=level_now, order='freq')
        if use_kurtosis:
            _ks = np.asarray([DenoiseMetricsHighOder().get(n.data, high_oder='kurtosis') for n in _nodes])
            _ks = np.abs(_ks)
            _ks = (_ks - np.min(_ks)) / (np.max(_ks) - np.min(_ks))
        else:
            _ks = np.ones(len(_nodes))

        _ss = np.asarray([DenoiseMetricsEntropy().get(n.data, entropy_type=entropy_type) for n in _nodes])

        if np.max(_ss) == np.min(_ss):
            _ses = np.ones(len(_nodes))
        else:
            _ses = (_ss - np.min(_ss)) / (np.max(_ss) - np.min(_ss))

        _vck = np.std(_ks) / np.mean(_ks)
        _vce = np.std(_ses) / (np.mean(_ses))

        # _scores = _ks * _ses
        # _score = np.mean(_scores)
        _score = _vce
        return _score


class DecomposeLevelSelectorClustering(DecomposeLevelSelectorABC):
    def __init__(self):
        super(DecomposeLevelSelectorClustering, self).__init__()

    def get_level_score(self, level_now, **kwargs):
        wavelet_inst = kwargs.get('wave_inst')

        _nodes = wavelet_inst.get_level(level=level_now, order='freq')

        pcs = []
        pcs_count = 0

        glength = len(wavelet_inst.data)

        for n in _nodes:
            d = n.data
            d = np.abs(d)
            length = len(d)

            delta = np.median(d) / 0.6745
            thd = delta * np.sqrt(2.0 * np.log(glength)) / np.log(level_now + 1)

            if np.max(d) < thd and False:
                continue
            else:
                pass
            d = (d - np.min(d)) / (np.max(d) - np.min(d) + np.finfo(d.dtype).eps)
            d = np.clip(d, 0.0, 1.0)
            if True:
                pc_max = np.log(0.5)
                pc = d * np.log(d + np.finfo(d.dtype).eps) + (1.0 - d) * np.log(1.0 - d + np.finfo(d.dtype).eps)
                pc /= np.log(level_now + 1)
            else:
                pc_max = 0.5
                pc = np.power(d, 2) + np.power(1.0 - d, 2)

            pc = np.mean(pc)

            pcs.append(pc)
        pcs = np.asarray(pcs)
        if len(pcs) == 0:
            pc = pc_max
        else:
            ratio = len(pcs) / (2 ** level_now)
            pc = np.mean(pcs) * len(pcs)
        return -pc


class DecomposeLevelSelectorFactory(object):
    def __init__(self):
        super(DecomposeLevelSelectorFactory, self).__init__()

    def create_selector(self, selector_type):
        if selector_type == 'preset':
            return DecomposeLevelSelectorPreset()

        if selector_type == 'entropy':
            return DecomposeLevelSelectorEntropy()

        if selector_type == 'clustering_pc':
            return DecomposeLevelSelectorClustering()


if __name__ == '__main__':
    pass
