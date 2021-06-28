import os
import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import kurtosis

from Cluster.CFuzzyClustering import CFuzzyCluster
from ImprovedWaveletPacketDenoise.DenoiseHelper import DenoiseMetricsEntropy
from ImprovedWaveletPacketDenoise.WaveletBasedNoiseSuppression import WaveletBasedNoiseSuppression
from MSPicker import DataPreparation
from Utils import SignalGenerator


class ComparisonABC(object):
    def __init__(self, path='.'):
        super(ComparisonABC, self).__init__()
        self.path = path
        self._mkdirs(path)

    @staticmethod
    def _mkdirs(path):
        try:
            os.makedirs(path)
        except:
            pass

    def if_not_exist_then_run(self, fname, func, **kwargs):
        overwrite = kwargs.get('overwrite', False)
        is_txt = kwargs.get('is_txt', True)

        fname = os.path.join(self.path, fname)
        if os.path.exists(fname) and not overwrite:
            if is_txt:
                data = np.loadtxt(fname)
            else:
                data = np.load(fname)
            return data

        data = func(**kwargs)
        if is_txt:
            np.savetxt(fname, data)
        else:
            np.save(fname, data)
        return data

    def exists(self, path):
        path = os.path.join(self.path, path)
        return os.path.exists(path)

    def load(self, fname, is_txt=True):
        if not fname.startswith(self.path):
            fname = os.path.join(self.path, fname)
        if not os.path.exists(fname):
            return None
        if is_txt:
            d = np.loadtxt(fname)
        else:
            d = np.load(fname)
        return d

    def save(self, data, fname, is_txt=True, overwrite=False):
        if not fname.startswith(self.path):
            fname = os.path.join(self.path, fname)
        if os.path.exists(data) and not overwrite:
            return
        if is_txt:
            np.savetxt(fname, data)
        else:
            np.save(fname, data)

    def load_noise(self, length, num, is_wgn):
        if not isinstance(is_wgn, str):
            is_wgn = 'wgn' if is_wgn else 'pink'
        nfname = '%d_%d_%s.npy' % (num, length, is_wgn)
        nfname = os.path.join(self.path, nfname)

        noise = self.load(nfname, is_txt=False)
        if noise is None:
            if is_wgn == 'wgn':
                noise = SignalGenerator.wgn(size=(num, length))
            else:
                noise = SignalGenerator.pink(size=(num, length))
            self.save(noise, nfname, is_txt=False, overwrite=True)
        return noise

    def compare(self, **kwargs):
        pass


class SynthesisDataComparison(ComparisonABC):
    def __init__(self):
        super(SynthesisDataComparison, self).__init__(path='SynthesisData')

    def compare(self, **kwargs):
        fm = kwargs.get('fm', 150)
        fs = kwargs.get('fs', 1000)
        length = kwargs.get('length', 1000)
        num = kwargs.get('num', 1000)
        snrs = kwargs.get('snrs', [10.0, 5.0, 1.0, -3.0, -5.0, -7.0, -8.0, -9.0, -10.0])
        noise_type = kwargs.get('noise_type', ['wgn', 'pink'])

        if fm > 0:
            sig = SignalGenerator.ricker(fm=fm, sample_rate=fs, length=length)
        else:
            sig = np.zeros(length)
            snrs = [0.0]

        fnames = ['K', 'P', 'Δ']
        wsize = 13

        tt = ['class', 'hard', 'soft', 'garrote']
        mother_wave = 'db8'
        level = 3

        ecount = 0

        def __get_score(thresholding_type, sn):
            _d = WaveletBasedNoiseSuppression().denoise(data=sn, wavelet=mother_wave, level=level,
                                                        decomposer_type='WaveletPacket',
                                                        decompose_level_selector_type='preset',
                                                        noise_estimator_type='global',
                                                        threshold_type='level_dependent',
                                                        thresholding_type=thresholding_type,
                                                        save_to=None,
                                                        fuzziness=2.0)

            features = DataPreparation(sn).get_features(fnames=fnames, wsize=wsize)
            cf = CFuzzyCluster()
            result = cf.fit(samples=features, n_cluster=2, fuzziness=2.0, operator_type='fcm',
                            init=np.asarray([[1.0] * len(fnames), [0.0] * len(fnames)]), initializer_type='manual')
            coeff = result.get_coefficients(min_in_binary=True, normalized=True)
            coeff = np.where(coeff < 0.25, coeff, 1.0)

            if fm > 0:
                if np.max(_d) == np.min(_d):
                    _corr = 0.0
                else:
                    _corr = np.corrcoef(sig, _d)[0][1]
                    _old_corr = np.abs(_corr)
            _old_rmse = np.std(sig - _d)

            _d = coeff * _d

            if fm > 0:
                if np.max(_d) == np.min(_d):
                    _corr = 0.0
                else:
                    _corr = np.corrcoef(sig, _d)[0][1]
                    _corr = np.abs(_corr)
            _rmse = np.std(sig - _d)
            if fm > 0:
                return [_old_corr, _old_rmse, _corr, _rmse]
            return [_old_rmse, _rmse]

        for nt in noise_type:
            noise = self.load_noise(num=num, length=length, is_wgn=nt)
            for snr in snrs:
                score_fname = '%0.4fHz_%dx%d_%dfs_%0.4fdB_%s_%s_%dL.txt' % (fm, num, length, fs, snr,
                                                                            nt, mother_wave, level)
                if self.exists(score_fname):
                    scores = self.load(score_fname, is_txt=True)
                else:
                    if fm > 0:
                        sns = SignalGenerator.signal_mixture(sig, noise, snr=snr)
                    else:
                        sns = np.copy(noise)
                    scores = None
                    for threshold_type in tt:
                        _scores = np.asarray([__get_score(threshold_type, sn) for sn in sns])
                        if scores is None:
                            scores = np.copy(_scores)
                        else:
                            scores = np.hstack((scores, _scores))
                    self.save(scores, score_fname, is_txt=True, overwrite=True)
                old_options = np.get_printoptions()
                np.set_printoptions(linewidth=np.inf)
                print('NT=%s, SNR=%0.4fdB: ' % (nt, snr), np.mean(scores, axis=0))
                np.set_printoptions(**old_options)


class RealFieldDataComparison(ComparisonABC):
    def __init__(self):
        super(RealFieldDataComparison, self).__init__(path='RealFieldData')

    @staticmethod
    def _sample_entropy(data):
        return DenoiseMetricsEntropy().get(data, entropy_type='sample_entropy', m=2, r=0.2)

    @staticmethod
    def _load_data():
        dset = pd.read_csv('mine_data.csv')
        samples = dset.to_numpy()
        labels = samples[:, -1]
        samples = samples[:, :-1]

        data_indexs = np.nonzero(labels > 0.0)[0]
        datas = samples[data_indexs, :]

        return datas

    def compare(self, **kwargs):
        noisy_traces = self._load_data()

        example_index = kwargs.get('example_index', 137)

        def __processing(**kwargs):
            def __filter(_sn, __tt):
                data = WaveletBasedNoiseSuppression().denoise(data=_sn, wavelet='db8', level=1,
                                                              decomposer_type='WaveletPacket',
                                                              decompose_level_selector_type='preset',
                                                              noise_estimator_type='global',
                                                              threshold_type='level_dependent',
                                                              thresholding_type=__tt,
                                                              save_to=None,
                                                              fuzziness=2.0)
                if __tt == 'class':
                    cf = CFuzzyCluster()
                    features = DataPreparation(_sn).get_features(wsize=13, fnames=['K', 'P', 'Δ'])
                    result = cf.fit(samples=features, n_cluster=2, fuzziness=2.0)
                    coeffs = result.get_coefficients(normalized=True, min_in_binary=True)
                    coeffs = np.where(coeffs < 0.25, coeffs, 1.0)
                    data = data * coeffs

                return data

            _data = kwargs.get('data')
            _tt = kwargs.get('tt')
            _pdata = np.asarray([__filter(_sn, _tt) for _sn in _data])
            return _pdata

        def __get_ses(**kwargs):
            _data = kwargs.get('data')
            _ses = np.asarray([self._sample_entropy(_sn) for _sn in _data])
            return _ses

        def __get_kurtosis(**kwargs):
            _data = kwargs.get('data')
            _ks = np.asarray([np.abs(kurtosis(_sn)) for _sn in _data])
            return _ks

        self.save(noisy_traces, 'noise_traces.txt', is_txt=True)

        noisy_traces_ses_fname = 'noisy_trace_ses.txt'
        noisy_ses = self.if_not_exist_then_run(noisy_traces_ses_fname, func=__get_ses,
                                               data=noisy_traces, is_txt=True)

        noisy_traces_kurtosis_fname = 'noisy_trace_kurtosis.txt'
        noisy_kurtosis = self.if_not_exist_then_run(noisy_traces_kurtosis_fname, func=__get_kurtosis,
                                                    data=noisy_traces, is_txt=True)

        print('Raw: ', np.mean(noisy_kurtosis), np.mean(noisy_ses))

        thresholding_types = ['hard', 'soft', 'garrote', 'class']
        for tt in thresholding_types:
            processed_traces_fname = 'processed_traces_' + tt
            processed_traces = self.if_not_exist_then_run(processed_traces_fname + '.txt', func=__processing,
                                                          data=noisy_traces, is_txt=True, tt=tt)

            processed_traces_ses_fname = processed_traces_fname + '_ses'
            processed_traces_ses = self.if_not_exist_then_run(processed_traces_ses_fname + '.txt', func=__get_ses,
                                                              data=processed_traces, is_txt=True)

            processed_traces_kurtosis_fname = processed_traces_fname + '_kurtosis'
            processed_kurtosis = self.if_not_exist_then_run(processed_traces_kurtosis_fname + '.txt',
                                                            func=__get_kurtosis, data=processed_traces, is_txt=True)

            print(tt, ': ', np.mean(processed_kurtosis), np.mean(processed_traces_ses))

            plt.figure()
            plt.plot(noisy_ses, label='Noisy')
            plt.plot(processed_traces_ses, label='Processed')
            plt.legend()

            plt.figure()
            plt.plot(noisy_kurtosis, label='Noisy')
            plt.plot(processed_kurtosis, label='Processed')
            plt.legend()

        plt.show()


class SynthesisDataComputationCostComparison(ComparisonABC):
    def __init__(self):
        super(SynthesisDataComputationCostComparison, self).__init__(path='SynthesisDataComputationCostComparison')

    def compare(self, **kwargs):
        fm = kwargs.get('fm', 150)
        fs = kwargs.get('fs', 1000)
        lengths = kwargs.get('length', [1000])
        num = kwargs.get('num', 1000)
        snr = kwargs.get('snr', -5.0)
        noise_type = kwargs.get('noise_type', ['wgn', 'pink'])

        fnames = ['K', 'P', 'Δ']
        wsize = 13

        tt = ['class', 'hard', 'soft', 'garrote']
        mother_wave = 'db8'
        level = 3

        def __get_times(sn):
            _times = []

            s_t = time.time()
            _d = WaveletBasedNoiseSuppression().denoise(data=sn, wavelet=mother_wave, level=level,
                                                        decomposer_type='WaveletPacket',
                                                        decompose_level_selector_type='preset',
                                                        noise_estimator_type='global',
                                                        threshold_type='level_dependent',
                                                        thresholding_type='class',
                                                        save_to=None,
                                                        fuzziness=2.0)
            features = DataPreparation(sn).get_features(fnames=fnames, wsize=wsize)
            cf = CFuzzyCluster()
            result = cf.fit(samples=features, n_cluster=2, fuzziness=2.0, operator_type='imfcm',
                            init=np.asarray([[1.0] * len(fnames), [0.0] * len(fnames)]),
                            initializer_type='manual', use_cuda=True)
            coeff = result.get_coefficients(min_in_binary=True, normalized=True)

            coeff = np.where(coeff < 0.25, coeff, 1.0)
            _d = _d * coeff

            e_t = time.time()

            _times.append(e_t - s_t)

            for thresholding_type in tt:
                s_t = time.time()
                _d = WaveletBasedNoiseSuppression().denoise(data=sn, wavelet=mother_wave, level=level,
                                                            decomposer_type='WaveletPacket',
                                                            decompose_level_selector_type='preset',
                                                            noise_estimator_type='global',
                                                            threshold_type='level_dependent',
                                                            thresholding_type=thresholding_type,
                                                            save_to=None,
                                                            fuzziness=2.0)
                if thresholding_type == 'class':
                    features = DataPreparation(sn).get_features(fnames=fnames, wsize=wsize)
                    cf = CFuzzyCluster()
                    result = cf.fit(samples=features, n_cluster=2, fuzziness=2.0, operator_type='imfcm',
                                    init=np.asarray([[1.0] * len(fnames), [0.0] * len(fnames)]),
                                    initializer_type='manual', use_cuda=False)
                    coeff = result.get_coefficients(min_in_binary=True, normalized=True)

                    coeff = np.where(coeff < 0.25, coeff, 1.0)
                    _d = _d * coeff

                e_t = time.time()

                _times.append(e_t - s_t)

            return _times

        for length in lengths:
            sig = SignalGenerator.ricker(fm=fm, sample_rate=fs, length=length)
            for nt in noise_type:
                noise = self.load_noise(num=num, length=length, is_wgn=nt)
                times_fname = '%0.4fHz_%dx%d_%dfs_%0.4fdB_%s_%s_%dL.txt' % (fm, num, length, fs, snr,
                                                                            nt, mother_wave, level)
                if self.exists(times_fname):
                    times = self.load(times_fname, is_txt=True)
                else:
                    sns = SignalGenerator.signal_mixture(sig, noise, snr=snr)
                    times = np.asarray([__get_times(sn) for sn in sns])
                    self.save(times, times_fname, is_txt=True, overwrite=True)
                old_options = np.get_printoptions()
                np.set_printoptions(linewidth=np.inf)
                print('NT=%s, Length=%d, SNR=%0.4fdB: ' % (nt, length, snr), np.mean(times, axis=0))
                np.set_printoptions(**old_options)


class ComputationCostAnalysis(ComparisonABC):
    def __init__(self):
        super(ComputationCostAnalysis, self).__init__(path='ComputationCostAnalysis')

    def compare(self, **kwargs):
        fm = kwargs.get('fm', 150)
        fs = kwargs.get('fs', 1000)
        lengths = kwargs.get('length', [1000])
        num = kwargs.get('num', 1000)
        snr = kwargs.get('snr', -5.0)

        fnames = ['K', 'P', 'Δ']
        wsize = 13

        tt = ['class', 'hard', 'soft', 'garrote']
        mother_wave = 'db8'
        level = 3

        for length in lengths:
            times = []

            sig = SignalGenerator.ricker(fm=fm, sample_rate=fs, length=length)
            noise = self.load_noise(length=length, num=1, is_wgn='wgn')

            sn = SignalGenerator.signal_mixture(sig, noise, snr=snr)[0]

            s_t = time.time()
            _d = WaveletBasedNoiseSuppression().denoise(data=sn, wavelet=mother_wave, level=level,
                                                        decomposer_type='WaveletPacket',
                                                        decompose_level_selector_type='preset',
                                                        noise_estimator_type='global',
                                                        threshold_type='level_dependent',
                                                        thresholding_type='class',
                                                        save_to=None,
                                                        fuzziness=2.0)
            times.append(time.time() - s_t)

            s_t = time.time()
            features = DataPreparation(sn).get_features(fnames=fnames, wsize=wsize)
            times.append(time.time() - s_t)

            s_t = time.time()
            cf = CFuzzyCluster()
            result = cf.fit(samples=features, n_cluster=2, fuzziness=2.0, operator_type='imfcm',
                            init=np.asarray([[1.0] * len(fnames), [0.0] * len(fnames)]),
                            initializer_type='manual', use_cuda=False)
            coeff = result.get_coefficients(min_in_binary=True, normalized=True)

            coeff = np.where(coeff < 0.25, coeff, 1.0)
            _d = _d * coeff
            times.append(time.time() - s_t)
            print('Length: ', length, ' Time: ', times, ', Total: ', sum(times))


if __name__ == '__main__':
    # Revision -1.0: Other test
    # Revision 0.0: 2021-05-24, first submitted
    # Revision 1.0: 2021-06-22, first revision

    Revision = 1.0

    if Revision == 1.0:
        SynthesisDataComputationCostComparison().compare(length=[100, 1000, 5000, 10000, 20000, 50000])

    if Revision == 0.0:
        SynthesisDataComparison().compare(fm=150)
        SynthesisDataComparison().compare(fm=0)

        RealFieldDataComparison().compare()

    if Revision == -1.0:
        ComputationCostAnalysis().compare(length=[100, 1000, 5000, 10000, 20000, 50000])
