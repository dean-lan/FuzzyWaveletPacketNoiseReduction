import numpy as np

import matplotlib.pyplot as plt

from itertools import permutations

from scipy.stats import kurtosis

from Cluster.CFuzzyClustering import CFuzzyCluster
from ImprovedWaveletPacketDenoise.DenoiseHelper import DenoiseMetricsEntropy
from MSPicker import DataPreparation


class ThresholdingABC(object):
    def __init__(self):
        super(ThresholdingABC, self).__init__()

    def apply(self, decomposition, thd, substitute=0.0, **kwargs):
        pass

    def draw(self, xrange=3.0, thd=1.0, **kwargs):
        num = kwargs.get('num', 100)
        _x = np.linspace(-xrange, xrange, num, endpoint=True)
        _y = self.apply(_x, thd, **kwargs)
        return _x, _y


class HardThresholding(ThresholdingABC):
    def __init__(self):
        super(HardThresholding, self).__init__()

    def apply(self, decomposition, thd, substitute=0.0, **kwargs):
        value = np.abs(decomposition)
        sign = np.sign(decomposition)
        pcoeff = sign * np.where(value < thd, substitute, value)
        return pcoeff


class SoftThresholding(ThresholdingABC):
    def __init__(self):
        super(SoftThresholding, self).__init__()

    def apply(self, decomposition, thd, substitute=0.0, **kwargs):
        value = np.abs(decomposition)
        sign = np.sign(decomposition)
        pcoeff = sign * np.where(value < thd, substitute, value - thd)
        return pcoeff


class GreaterThresholding(ThresholdingABC):
    def __init__(self):
        super(GreaterThresholding, self).__init__()

    def apply(self, decomposition, thd, substitute=0.0, **kwargs):
        pcoeff = np.where(decomposition < thd, substitute, decomposition)
        return pcoeff


class LessThresholding(ThresholdingABC):
    def __init__(self):
        super(LessThresholding, self).__init__()

    def apply(self, decomposition, thd, substitute=0.0, **kwargs):
        pcoeff = np.where(decomposition > thd, substitute, decomposition)
        return pcoeff


class GarroteThresholding(ThresholdingABC):
    def __init__(self):
        super(GarroteThresholding, self).__init__()

    def apply(self, decomposition, thd, substitute=0.0, **kwargs):
        pthreshold = np.power(thd, 2.0)
        value = np.abs(decomposition)
        sign = np.sign(decomposition)
        value = np.where(value < thd, substitute, value - pthreshold / value)
        pcoeff = sign * value
        return pcoeff


class RatioThresholding(ThresholdingABC):
    def __init__(self):
        super(RatioThresholding, self).__init__()
        self.decomposition_inst_now = None
        self.alphas = None

    @staticmethod
    def __entropy(data, **kwargs):
        entropy_type = kwargs.get('metric')
        entropy_metrics = DenoiseMetricsEntropy()
        return entropy_metrics.get(data, entropy_type=entropy_type, **kwargs)

    def _get_metric_func(self, metric_type):
        metric_type = metric_type.lower()
        if 'entropy' in metric_type:
            return self.__entropy

    def _cal_alphas(self, **kwargs):
        decompositions = kwargs.get('decompositions')
        metric = kwargs.get('metric', 'sample_entropy')
        kwargs['metric'] = metric

        metric_func = self._get_metric_func(metric)
        entropy = np.asarray([metric_func(d, **kwargs) for d in decompositions])

        alphas = (entropy - np.min(entropy)) / (np.max(entropy) - np.min(entropy))

        # metric_func = self._get_metric_func('fuzzy_entropy')
        # kwargs['metric'] = 'fuzzy_entropy'
        # fe_entropy = np.asarray([metric_func(d, **kwargs) for d in decompositions])
        # fe_alphas = (fe_entropy - np.min(fe_entropy)) / (np.max(fe_entropy) - np.min(fe_entropy))
        #
        # print(fe_alphas, alphas)

        self.alphas = alphas

    def apply(self, decomposition, thd, substitute=0.0, **kwargs):
        index = kwargs.get('index')
        decompositions = kwargs.get('decompositions')

        if self.decomposition_inst_now != decompositions:
            self.decomposition_inst_now = decompositions
            self._cal_alphas(**kwargs)

        value = np.abs(decomposition)
        sign = np.sign(decomposition)
        alpha = self.alphas[index]
        old_options = np.geterr()
        np.seterr(all='raise')

        if alpha < np.finfo(alpha.dtype).eps:
            thou = np.zeros(len(decomposition))
        else:
            thou = np.exp(-(value - thd) * (1.0 - alpha) / alpha)
        np.seterr(**old_options)
        pcoeff = sign * np.where(value < thd, 0.0, value - thd * thou)
        return pcoeff


class ClassificationThresholding(RatioThresholding):
    def __init__(self):
        super(ClassificationThresholding, self).__init__()
        self.decomposition_inst_now = None
        self.alphas = None
        self.timed_coeffs = None
        self.timed_labels = None

    def _get_signal_labels(self, decomposition, thd, **kwargs):
        fuzziness = kwargs.get('fuzziness', 2.0)
        wsize = kwargs.get('wsize', 13)
        fnames = kwargs.get('fnames', ['K', 'P', 'Δ'])

        # 1. 直接通过阈值找到主瓣位置，进行AIC校正；+++ Try this first.
        # 2. 第二种方法为，通过聚类后进行相应操作，但是否以给定阈值在聚类中的隶属度系数作为划分的系数阈值进行分类？

        # featuers = DataPreparation(decomposition).get_features(fnames=fnames, wsize=wsize, standard='MinMax')
        # cf = CFuzzyCluster()
        # result = cf.fit(samples=featuers, n_cluster=2, fuzziness=fuzziness, operator_type='imfcm',
        #                 init=np.asarray([[1.0] * len(fnames), [0.0] * len(fnames)]), initializer_type='manual')
        # coeffs = result.get_coefficients(min_in_binary=True, normalized=True)
        thd = np.std(decomposition) * 0.2
        coeffs = 1.0 - np.exp(-np.power(np.abs(decomposition), fuzziness) / thd)
        coeffs = np.clip(coeffs, 0.0, 1.0)

        abs_dep = np.abs(decomposition)
        # coeffs = np.where(abs_dep > thd, 1.0, (1.0 - np.exp(-np.power(abs_dep, fuzziness) / thd)))
        coeffs = 1.0 - np.exp(-np.power(abs_dep, fuzziness) / thd)
        # coeffs = np.where(coeffs < 0.25, coeffs, 1.0)
        return coeffs

    def apply(self, decomposition, thd, substitute=0.0, **kwargs):
        index = kwargs.get('index')
        decompositions = kwargs.get('decompositions')

        if np.max(np.abs(decomposition)) < thd:
            return np.zeros(len(decomposition))

        coeffs = self._get_signal_labels(decomposition, thd, **kwargs)

        pcoeff = decomposition * coeffs
        return pcoeff


class FuzzyThresholding(ThresholdingABC):
    def __init__(self):
        super(FuzzyThresholding, self).__init__()

    def apply(self, decomposition, thd, substitute=0.0, **kwargs):
        alpha = kwargs.get('alpha', 1.0)
        r = kwargs.get('r', 0.01)
        fuzziness = kwargs.get('fuzziness', 2.0)

        delta = np.std(decomposition)
        abs_coeff = np.abs(decomposition)

        s = np.exp(alpha *
                         np.power(abs_coeff - thd, fuzziness) / (r * delta + np.finfo(delta.dtype).eps))

        s = np.clip(s, 0.0, 1.0)
        # fuzzy_coeffs = np.exp(-(abs_coeff - thd) * (1.0 - s) / (s + np.finfo(s.dtype).eps))

        sign = np.sign(decomposition)
        pcoeff = sign * abs_coeff * s
        return pcoeff


class FuzzyRatioThresholding(RatioThresholding):
    def __init__(self):
        super(FuzzyRatioThresholding, self).__init__()

    def apply(self, decomposition, thd, substitute=0.0, **kwargs):
        pass


class ThresholdingFactory(object):
    def __init__(self):
        super(ThresholdingFactory, self).__init__()

    def create_thresholding(self, thresholding_type='garrote'):
        if thresholding_type == 'hard':
            return HardThresholding()

        if thresholding_type == 'soft':
            return SoftThresholding()

        if thresholding_type == 'garrote':
            return GarroteThresholding()

        if thresholding_type == 'ratio':
            return RatioThresholding()

        if thresholding_type == 'fuzzy':
            return FuzzyThresholding()

        return ClassificationThresholding()


if __name__ == '__main__':
    coeff = np.random.randn(500)
    threshold = 1.0
    GT = ThresholdingFactory().create_thresholding(thresholding_type='fuzzy')
    fuzziness = np.linspace(2.0, 3.0, 1, endpoint=False)
    plt.figure()
    _x, _y = ThresholdingFactory().create_thresholding(thresholding_type='hard').draw(xrange=4)
    plt.plot(_x, _y, label='Hard')
    _x, _y = ThresholdingFactory().create_thresholding(thresholding_type='soft').draw(xrange=4)
    plt.plot(_x, _y, label='Soft')

    for f in fuzziness:
        _x, _y = GT.draw(alpha=1.0, r=0.01, fuzziness=f, xrange=4, num=1000)
        plt.plot(_x, _y, label='fuzziness=%0.1f' % f)
    plt.legend()
    plt.show()
