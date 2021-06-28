import numpy as np


class ThresholdValueABC(object):
    def __init__(self):
        super(ThresholdValueABC, self).__init__()

    def get(self, delta, glength, **kwargs):
        g_threshold = delta * np.sqrt(2.0 * np.log(glength))
        return g_threshold


class TraditionalThreshold(ThresholdValueABC):
    def __init__(self):
        super(TraditionalThreshold, self).__init__()


class WaveletPacketSpecifiedThreshold(ThresholdValueABC):
    def __init__(self):
        super(WaveletPacketSpecifiedThreshold, self).__init__()

    def get(self, delta, glength, **kwargs):
        thd = delta * np.sqrt(2.0 * glength * np.log2(glength))
        return thd


class LevelDependentThreshold(ThresholdValueABC):
    def __init__(self):
        super(LevelDependentThreshold, self).__init__()

    def get(self, delta, glength, **kwargs):
        level = kwargs.get('level')
        thd = delta * np.sqrt(2.0 * np.log(glength)) / np.log(level + 1)
        return thd


class NodeDependentThreshold(ThresholdValueABC):
    def __init__(self):
        super(NodeDependentThreshold, self).__init__()

    def get(self, delta, glength, **kwargs):
        thd = delta * np.sqrt(2 * glength)
        return thd


class ThresholdFactory(object):
    def __init__(self):
        super(ThresholdFactory, self).__init__()

    def create_threshold(self, threshold_type):
        if threshold_type == 'traditional':
            return TraditionalThreshold()

        if threshold_type == 'level_dependent':
            return LevelDependentThreshold()

        if threshold_type == 'node_dependent':
            return NodeDependentThreshold()

        if threshold_type == 'waveletpacketspecified':
            return WaveletPacketSpecifiedThreshold()
