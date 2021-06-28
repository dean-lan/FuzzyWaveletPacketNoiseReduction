import os

import pywt

import PyEMD

import numpy as np

import matplotlib.pyplot as plt

from ImprovedWaveletPacketDenoise.DenoiseHelper import DenoiseHelper, DenoiseMetricsEntropy
from Utils import SignalGenerator


class DecomposerABC(object):
    def __init__(self):
        super(DecomposerABC, self).__init__()
        self.wavelet = None
        self.level = 0
        self.decompositions = None

    def __getitem__(self, key):
        raise Exception('Null implement of getitem for decomposer')

    def __setitem__(self, key, value):
        raise Exception('Null implement of setitem for decomposer')

    def __len__(self):
        raise Exception('Null implement of len() for decomposer')

    def get(self, index, level=None):
        level = self.level if level is None or level < 0 else level
        old_level = self.level
        self.level = level
        d = self[index]
        self.level = old_level
        return d

    def set(self, index, data, level=None):
        level = self.level if level is None or level < 0 else level
        old_level = self.level
        self.level = level
        self[index] = data
        self.level = old_level

    def decompose(self, data, **kwargs):
        pass

    def recovery(self):
        pass

    def show(self):
        pass

    def save(self, save_to='.', suffix=None):
        if self.decompositions is None:
            return
        DenoiseHelper().mkdirs(save_to)
        for _i in range(len(self)):
            d = self[_i]
            if suffix is None:
                fname = os.path.join(save_to, '%d.txt' % _i)
            else:
                fname = os.path.join(save_to, '%d_%s.txt' % (_i, suffix))
            np.savetxt(fname, d)

    def do_save(self, **kwargs):
        save_to = kwargs.get('save_to', None)
        suffix = kwargs.get('suffix', None)
        if save_to is None:
            return
        if not isinstance(save_to, str):
            print('can\'t save to path given not as a string')
            return
        DenoiseHelper().mkdirs(save_to)
        self.save(save_to=save_to, suffix=suffix)


class DataDecomposerWavelet(DecomposerABC):
    def __init__(self):
        super(DataDecomposerWavelet, self).__init__()

    def __getitem__(self, key):
        if self.decompositions is None or key >= len(self.decompositions):
            raise StopIteration()

        return self.decompositions[key]

    def __setitem__(self, key, value):
        self.decompositions[key] = value

    def __len__(self):
        return len(self.decompositions)

    def decompose(self, data, **kwargs):
        wavelet = kwargs.get('wavelet')
        level = kwargs.get('level')

        wave_inst = pywt.Wavelet(wavelet)
        maxlevel = pywt.dwt_max_level(len(data), wave_inst)
        level = maxlevel if level is None else np.min((level, maxlevel))

        self.wavelet = wave_inst
        self.level = level

        self.decompositions = pywt.wavedec(data, wavelet, level=level)

    def recovery(self, **kwargs):
        data = pywt.waverec(self.decompositions, wavelet=self.wavelet)
        data = np.copy(data)
        return data


class DataDecomposerWaveletPacket(DecomposerABC):
    def __init__(self):
        super(DataDecomposerWaveletPacket, self).__init__()

    def __getitem__(self, key):
        length = 2 ** self.level
        if self.decompositions is None or key >= length:
            raise StopIteration()
        nodes = self.decompositions.get_level(self.level, order='freq')
        data = np.copy(nodes[key].data)
        return data

    def __setitem__(self, key, value):
        nodes = self.decompositions.get_level(self.level, order='freq')
        nodes[key].data = value

    def __len__(self):
        return 2 ** self.level

    def decompose(self, data, **kwargs):
        wavelet = kwargs.get('wavelet', 'db10')
        level = kwargs.get('level', None)

        wave_inst = pywt.WaveletPacket(data, wavelet)
        maxlevel = wave_inst.maxlevel
        level = maxlevel if level is None else np.min((level, maxlevel))

        self.decompositions = wave_inst
        self.wavelet = wavelet
        self.level = level

    def recovery(self):
        data = self.decompositions.reconstruct()
        return data


class CEEMDANDecomposer(DecomposerABC):
    def __init__(self):
        super(CEEMDANDecomposer, self).__init__()

    def __getitem__(self, key):
        length = len(self)
        if self.decompositions is None or ((key > length) if not isinstance(key, (tuple, list, np.ndarray)) else (max(key) > length)):
            raise StopIteration()
        data = np.copy(self.decompositions[key])
        return data

    def __setitem__(self, key, value):
        if self.decompositions is None:
            return
        self.decompositions[key] = value

    def __len__(self):
        return 0 if self.decompositions is None else len(self.decompositions[:, 0])

    def decompose(self, data, **kwargs):
        ceemdan = PyEMD.CEEMDAN(parallel=True)
        imfs = ceemdan(data)

        self.wavelet = ceemdan
        self.decompositions = imfs

    def recovery(self):
        data = np.asarray(self.decompositions)
        data = np.sum(data, axis=0)
        return data

    def show(self):
        vis = PyEMD.Visualisation()
        vis.plot_imfs(imfs=self.decompositions, residue=None, include_residue=False)
        vis.show()


class DecomposerFactory(object):
    def __init__(self):
        super(DecomposerFactory, self).__init__()

    def create_decomposer(self, composer_type):
        composer_type = composer_type.lower()
        if composer_type == 'wavelet':
            return DataDecomposerWavelet()

        if composer_type == 'waveletpacket':
            return DataDecomposerWaveletPacket()

        if composer_type == 'ceemdan':
            return CEEMDANDecomposer()


def test():
    ecg = pywt.data.ecg()

    wp = pywt.WaveletPacket(ecg, 'sym5', maxlevel=4)

    fig = plt.figure()
    plt.set_cmap('bone')
    ax = fig.add_subplot(wp.maxlevel + 1, 1, 1)
    ax.plot(ecg, 'k')
    ax.set_xlim(0, len(ecg) - 1)
    ax.set_title("Wavelet packet coefficients")

    for level in range(1, wp.maxlevel + 1):
        ax = fig.add_subplot(wp.maxlevel + 1, 1, level + 1)
        nodes = wp.get_level(level, "freq")
        nodes.reverse()
        labels = [n.path for n in nodes]
        values = -abs(np.array([n.data for n in nodes]))
        ax.imshow(values, interpolation='nearest', aspect='auto')
        ax.set_yticks(np.arange(len(labels) - 0.5, -0.5, -1), labels)
        plt.setp(ax.get_xticklabels(), visible=False)

    plt.show()


if __name__ == '__main__':
    fm = 150
    fs = 1000
    length = 512

    snr = -5.0

    sig = SignalGenerator.ricker(fm=fm, sample_rate=fs, length=length)
    noise = SignalGenerator.wgn(size=length)

    sn = SignalGenerator.signal_mixture(sig, noise, snr)

    # test()

    decomposer = DataDecomposerWaveletPacket()
    decomposer.decompose(data=sn, wavelet='db8', level=None)

    entropy = DenoiseMetricsEntropy()
    se = entropy.get(sn, entropy_type='sample_entropy')
    fe = entropy.get(sn, entropy_type='fuzzy_entropy')
    print('Level: ', 0)
    print('Sample Entropy: ', se)
    print('Fuzzy Entropy: ', fe)

    for level in range(1, decomposer.level + 1):
        node_num = 2 ** level
        print('Level: ', level)
        ses = []
        fes = []
        for j in range(node_num):
            d = decomposer.get(j, level=level)
            entropy = DenoiseMetricsEntropy()
            ses.append(entropy.get(d, entropy_type='sample_entropy'))
            fes.append(entropy.get(d, entropy_type='fuzzy_entropy'))
        print('Sample Entropy: ', ses)
        print('Fuzzy Entropy: ', fes)
        print()
    print()

    decomposer = CEEMDANDecomposer()
    decomposer.decompose(sn)

    pcs = np.asarray([np.corrcoef(d, sn)[0][1] for d in decomposer])
    ses = np.asarray([DenoiseMetricsEntropy().get(d, entropy_type='sample_entropy') for d in decomposer])

    fes = np.asarray([DenoiseMetricsEntropy().get(d, entropy_type='fuzzy_entropy') for d in decomposer])

    pcs_index = np.nonzero(pcs < 0.2)[0]
    mc = np.sum(decomposer[pcs_index], axis=0)
    pcs_index = len(decomposer) - 1 - pcs_index
    minor_conponent = np.sum(decomposer[pcs_index], axis=0)

    plt.figure()
    plt.subplot(211)
    plt.plot(mc)
    plt.subplot(212)
    plt.plot(minor_conponent)

    plt.figure()
    plt.subplot(211)
    plt.plot(sig)
    plt.subplot(212)
    plt.plot(sn)

    plt.figure()
    plt.plot(ses, label='Sample Entropy')
    plt.legend()

    plt.figure()
    plt.plot(fes, label='Fuzzy Entropy')
    plt.legend()

    plt.figure()
    plt.plot(pcs, label='Corr')
    plt.legend()

    decomposer.show()










