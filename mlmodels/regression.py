import numpy as np
from matplotlib import pyplot as plt
from mlutils.modelfuncs import ModelFuncs


class RegressionFuncs(ModelFuncs):

    def loss(self, pv, dataset):
        return self.model.f_loss(pv, dataset.input, dataset.target)

    def loss_grad(self, pv, dataset):
        return self.model.f_loss_grad(pv, dataset.input, dataset.target)

    def predict(self, pv, dataset):
        return self.model.f_predict(pv, dataset.input)


class SequenceRegressionFuncs(ModelFuncs):

    def loss(self, pv, dataset):
        return self.model.f_loss(pv, dataset.valid, dataset.input, dataset.target)

    def loss_grad(self, pv, dataset):
        return self.model.f_loss_grad(pv, dataset.valid, dataset.input, dataset.target)

    def predict(self, pv, dataset):
        return self.model.f_predict(pv, dataset.input)

