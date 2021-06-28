import os

import numpy as np
from scipy.stats import kurtosis, skew

from Utils import SignalGenerator


class DenoiseHelper(object):
    def __init__(self, path='.'):
        super(DenoiseHelper, self).__init__()
        self.path = path

        self.mkdirs(self.path)

    @staticmethod
    def to_range(data, low=0.0, high=1.0):
        d = np.atleast_2d(data)
        vmin = np.min(d, axis=1, keepdims=True)
        vmax = np.max(d, axis=1, keepdims=True)
        d = (d - vmin) / (vmax - vmin)

        d = (high - low) * d + low

        if len(data.shape) == 1 or data.shape[0] == 1:
            d = d[0]

        return d

    @staticmethod
    def mkdirs(dir_name):
        try:
            os.makedirs(dir_name)
        except:
            pass

    def save(self, fname, data, **kwargs):
        overwrite = kwargs.get('overwrite', True)
        is_txt = kwargs.get('is_txt', False)

        fpath = os.path.join(self.path, fname)

        if os.path.exists(fpath) and not overwrite:
            return

        if is_txt:
            np.savetxt(fpath, data)
        else:
            np.save(fpath, data)

    def load(self, fname, **kwargs):
        is_txt = kwargs.get('is_txt', False)
        fpath = os.path.join(self.path, fname)

        if not os.path.exists(fpath):
            return None
        if is_txt:
            data = np.loadtxt(fpath)
        else:
            data = np.load(fpath)
        return data

    def if_not_exist_then_run(self, fname, func, **kwargs):
        data = self.load(fname)
        if data is None:
            data = func(**kwargs)
        return data

    def load_noise(self, **kwargs):
        num = kwargs.get('num', 1000)
        length = kwargs.get('length', 500)
        is_wgn = kwargs.get('is_wgn', True)
        is_txt = kwargs.get('is_txt', False)
        do_save = kwargs.get('do_save', True)

        if not isinstance(is_wgn, str):
            is_wgn = 'wgn' if is_wgn is True else 'pink'

        nfname = '%dx%d_%s' % (num, length, is_wgn)
        nfname += '.txt' if is_txt else '.npy'

        def __create_noise(**kwargs):
            if is_wgn == 'wgn':
                _noise = SignalGenerator.wgn(size=(num, length))
            else:
                _noise = SignalGenerator.pink(size=(num, length))
            if do_save:
                self.save(nfname, _noise)
            return _noise

        noise = self.if_not_exist_then_run(nfname, __create_noise, **kwargs)
        return noise


class DenoiseMetricsABC(object):
    def __init__(self):
        super(DenoiseMetricsABC, self).__init__()

    def get(self, data, **kwargs):
        pass


class DenoiseMetricsEntropy(DenoiseMetricsABC):
    def __init__(self):
        super(DenoiseMetricsEntropy, self).__init__()

    @staticmethod
    def __DIST(data, **kwargs):
        _m = kwargs.get('m')
        _slice_preprocess = kwargs.get('slice_preprocess', None)

        _n = len(data)

        n = _n
        m = _m

        _k = n - _m + 1
        _kindex = np.linspace(0, _k, _k, endpoint=False, dtype=np.int)
        _slices = np.asarray([data[_i: _i + m] for _i in range(_k)])
        if _slice_preprocess is not None:
            _slices = np.asarray([_slice_preprocess(_s) for _s in _slices])
        _distances = np.zeros(shape=(_k, _k))
        for _i in range(_k):
            _slices_diff = _slices - _slices[_i]
            _abs_ij = np.abs(_slices_diff)
            _dij = np.max(_abs_ij, axis=1)
            _distances[_i, :] = _dij
        return _distances

    @staticmethod
    def __SAMPLE_ENTROPY_PHI(data, **kwargs):
        _n = len(data)
        _m = kwargs.get('m')

        n = _n
        m = _m

        _r = kwargs.get('r', 0.2)
        _sd = np.std(data)
        _f = _r * _sd
        _distances = DenoiseMetricsEntropy.__DIST(data, **kwargs)
        _counts = np.where(_distances > _f, 0.0, 1.0)
        _count_sum = np.sum(_counts, axis=1) - 1.0  # remove the situation of i == j
        _count_ratio = _count_sum / (n - m)
        _phi = np.sum(_count_ratio) / (n - m + 1)
        return _phi

    @staticmethod
    def __APPROXIMATE_ENTROPY_PHI(data, **kwargs):
        _n = len(data)
        _m = kwargs.get('m')

        n = _n
        m = _m

        _r = kwargs.get('r', 0.2)
        _sd = np.std(data)
        _f = _r * _sd
        _distances = DenoiseMetricsEntropy.__DIST(data, **kwargs)
        _counts = np.where(_distances > _f, 0.0, 1.0)
        _count_sum = np.sum(_counts, axis=1)
        _count_ratio = _count_sum / (n - m + 1)
        _phi = np.mean(_count_ratio)
        return _phi

    @staticmethod
    def __FUZZY_ENTROPY_PHI(data, **kwargs):
        _n = len(data)
        _m = kwargs.get('m')

        n = _n
        m = _m

        _r = kwargs.get('r', 0.2)
        _fuzziness = kwargs.get('fuzziness', 2.0)
        _sd = np.std(data)
        _f = _r * _sd

        def __slice_preprocess(_slice):
            return _slice - np.mean(_slice)

        _distances = DenoiseMetricsEntropy.__DIST(data, slice_preprocess=__slice_preprocess, **kwargs)
        _degrees = np.exp(-np.power(_distances, _fuzziness) / _f)
        _mean_degree = (np.sum(_degrees, axis=1) - 1.0) / (n - m - 1)
        _phi = np.sum(_mean_degree) / (n - m)
        return _phi

    def _sample_entropy(self, data, **kwargs):
        m = kwargs.get('m', 2)

        kwargs['m'] = m

        phi_m = self.__SAMPLE_ENTROPY_PHI(data, **kwargs)

        kwargs['m'] = m + 1
        phi_mp1 = self.__SAMPLE_ENTROPY_PHI(data, **kwargs)

        return np.log(phi_m + np.finfo(phi_m.dtype).eps) - np.log(phi_mp1 + np.finfo(phi_m.dtype).eps)

    def _approximate_entropy(self, data, **kwargs):
        m = kwargs.get('m', 2)

        kwargs['m'] = m
        phi_m = self.__APPROXIMATE_ENTROPY_PHI(data, **kwargs)

        kwargs['m'] = m + 1
        phi_mp1 = self.__APPROXIMATE_ENTROPY_PHI(data, **kwargs)

        return phi_m - phi_mp1

    def _fuzzy_entropy(self, data, **kwargs):
        m = kwargs.get('m', 2)

        kwargs['m'] = m
        phi_m = self.__FUZZY_ENTROPY_PHI(data, **kwargs)

        kwargs['m'] = m + 1
        phi_mp1 = self.__FUZZY_ENTROPY_PHI(data, **kwargs)

        return np.log(phi_m / (phi_mp1 + np.finfo(phi_m.dtype).eps))

    def get(self, data, **kwargs):
        entropy_type = kwargs.get('entropy_type', 'sample_entropy')

        if np.max(data) == np.min(data):
            return 0.0

        if 'fuzzy' in entropy_type:
            return self._fuzzy_entropy(data, **kwargs)
        if 'sample' in entropy_type:
            return self._sample_entropy(data, **kwargs)
        if 'approximate' in entropy_type:
            return self._approximate_entropy(data, **kwargs)


class DenoiseMetricsVarianceCoefficient(DenoiseMetricsABC):
    def __init__(self):
        super(DenoiseMetricsVarianceCoefficient, self).__init__()

    def get(self, data, **kwargs):
        _delta = np.std(data)
        _m = np.mean(data)
        return _delta / (_m + np.finfo(_delta.dtype).eps)


class DenoiseMetricsHighOder(DenoiseMetricsABC):
    def __init__(self):
        super(DenoiseMetricsHighOder, self).__init__()

    def get(self, data, **kwargs):
        high_order = kwargs.get('high_oder', 'kurtosis')

        if high_order == 'kurtosis':
            return kurtosis(data)
        if high_order == 'skewness':
            return skew(data)


if __name__ == '__main__':
    pass
