import numpy as np


class NoiseEstimatorABC(object):
    def __init__(self):
        super(NoiseEstimatorABC, self).__init__()

    def get(self, decomposer, **kwargs):
        pass


class GlobalNoiseEstimator(NoiseEstimatorABC):
    def __init__(self):
        super(GlobalNoiseEstimator, self).__init__()

    def get(self, decomposer, **kwargs):
        coeff = decomposer[-1]
        abs_coeff = np.abs(coeff)
        delta = np.median(abs_coeff) / 0.6745
        return delta


class LevelDependentNoiseEstimator(NoiseEstimatorABC):
    def __init__(self):
        super(LevelDependentNoiseEstimator, self).__init__()

    def get(self, decomposer, **kwargs):
        coeff = decomposer.get(-1, 1)
        abs_coeff = np.abs(coeff)
        delta = np.median(abs_coeff) / 0.6745
        return delta


class NodeDependentNoiseEstimator(NoiseEstimatorABC):
    def __init__(self):
        super(NodeDependentNoiseEstimator, self).__init__()

    def get(self, decomposer, **kwargs):
        index = kwargs.get('index')
        coeff = decomposer[index]
        abs_coeff = np.abs(coeff)
        delta = np.median(abs_coeff) / 0.6745
        return delta


class NoiseEstimatorFactory(object):
    def __init__(self):
        super(NoiseEstimatorFactory, self).__init__()

    def create_estimator(self, estimator_type):
        if estimator_type == 'normal' or estimator_type == 'global':
            return GlobalNoiseEstimator()

        if estimator_type == 'level_dependent':
            return LevelDependentNoiseEstimator()

        if estimator_type == 'node_dependent':
            return NodeDependentNoiseEstimator()


