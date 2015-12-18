from os.path import join

import numpy as np
from matplotlib import pyplot as plt

from mlutils import load_cfg, gather, xp, pca_white
from mlutils.modelfuncs import ModelFuncs
from mlutils.plot import plot_weight_histograms
from mlutils.preprocess import pca_white_inverse


default_cfg = {'func_class': None,
               'dataset_samples': None,
               'dataset_input': 'input',
               'dataset_target': 'target',
               'max_missed_val_improvements': 100,
               'iteration_gain': 0,
               'preprocess_pca': None,
               'use_training_as_validation': False,
               'no_negative_data': False,
               'positive_weights_init': False,
               'minibatch_size': 200
               }


class FeedForwardFuncs(ModelFuncs):

    def __init__(self, cfg):
        if cfg.preprocess_pca is not None and cfg.preprocess_pca != cfg.n_units[0]:
            raise ValueError("number of PCA components must match input unit count")
        if cfg.preprocess_pca is not None and cfg.loss == 'cross_entropy':
            raise ValueError("PCA whitening does not work with cross entropy loss")
        super(FeedForwardFuncs, self).__init__(self.create_model(cfg), cfg)

    def create_model(self, cfg):
        """
        Creates the employed model.
        :param cfg: configuration
        :return: the model
        """
        return cfg.model
        # return MLP(cfg.loss, cfg.n_units, cfg.transfer_func, cfg)

    @classmethod
    def train_from_cfg(cls):
        """
        Creates and trains model functions using configuration specified at command line.
        """
        # create model
        cfg, cfg_dir, cph, cp = load_cfg(defaults=default_cfg, with_checkpoint=True)
        if cfg.func_class is not None:
            funcs = cfg.func_class(cfg)
        else:
            funcs = cls(cfg)

        # train
        his = funcs.generic_training(cfg_dir, cp, cph,
                                     max_missed_val_improvements=cfg.max_missed_val_improvements,
                                     iteration_gain=cfg.iteration_gain)

        # plot weight histogram
        plt.figure(figsize=(14, 14))
        plot_weight_histograms(funcs.ps, funcs.ps.all_vars())
        plt.savefig(join(cfg_dir, "weights.pdf"))
        plt.close()

        # obtain predictions
        trn_inp = gather(funcs.dataset.trn.input)
        trn_tgt = gather(funcs.dataset.trn.target)
        trn_pred = gather(funcs.predict(funcs.ps.data, funcs.dataset.trn))
        funcs.show_results('trn', funcs.dataset.trn, trn_inp, trn_tgt, trn_pred)

        tst_inp = gather(funcs.dataset.tst.input)
        tst_tgt = gather(funcs.dataset.tst.target)
        tst_pred = gather(funcs.predict(funcs.ps.data, funcs.dataset.tst))
        funcs.show_results('tst', funcs.dataset.tst, tst_inp, tst_tgt, tst_pred)

        return dict(cfg=cfg, cfg_dir=cfg_dir, funcs=funcs, his=his,
                    trn_inp=trn_inp, trn_tgt=trn_tgt, trn_pred=trn_pred,
                    tst_inp=tst_inp, tst_tgt=tst_tgt, tst_pred=tst_pred)

    def show_results(self, partition, ds, inp, tgt, pred):
        pass

    def loss(self, pv, dataset):
        return self.model.f_loss(pv, dataset.input, dataset.target)

    def loss_grad(self, pv, dataset):
        return self.model.f_loss_grad(pv, dataset.input, dataset.target)

    def predict(self, pv, dataset):
        return self.model.f_predict(pv, dataset.input)

    def init_parameters(self):
        super(FeedForwardFuncs, self).init_parameters()

        if self.cfg.positive_weights_init:
            print "Ensuring positive initialization weights."
            self.ps.data[:] = xp.abs(self.ps.data)

    def preprocess_dataset(self, ds):
        if self.cfg.dataset_input != 'input':
            ds['input'] = ds[self.cfg.dataset_input]
            del ds[self.cfg.dataset_input]
        if self.cfg.dataset_target != 'target':
            ds['target'] = ds[self.cfg.dataset_target]
            del ds[self.cfg.dataset_target]

        if self.cfg.dataset_samples is not None:
            ds['input'] = ds['input'][..., 0:self.cfg.dataset_samples]
            ds['target'] = ds['target'][..., 0:self.cfg.dataset_samples]
            print "Using only %d samples from dataset" % ds['input'].shape[-1]

        if self.cfg.no_negative_data:
            minval = np.min(ds['input'])
            if minval < 0:
                print "Adding %.3f to dataset inputs to ensure positive values." % (-minval)
                ds['input'] -= minval
            else:
                print "Dataset inputs are already positive."

        if self.cfg.preprocess_pca is not None:
            ds['orig_input'] = np.copy(ds['input'])
            ds['input'], ds['meta_pca_vars'], ds['meta_pca_axes'], ds['meta_pca_means'] = \
                pca_white(ds['input'], n_components=self.cfg.preprocess_pca, return_axes=True)
            print "Keeping %d principal components (PCA) with variances:" % self.cfg.preprocess_pca
            print ds['meta_pca_vars']
            np.savez_compressed(join(self.cfg.out_dir, "pca.npz"),
                                pca_vars=ds['meta_pca_vars'],
                                pca_axes=ds['meta_pca_axes'],
                                pca_means=ds['meta_pca_means'])

        if self.cfg.use_training_as_validation:
            ds['meta_use_training_as_validation'] = self.cfg.use_training_as_validation

        return ds

    def invert_pca(self, whitened):
        if self.cfg.preprocess_pca is not None:
            return pca_white_inverse(whitened,
                                     self.dataset.meta_pca_vars,
                                     self.dataset.meta_pca_axes,
                                     self.dataset.meta_pca_means)
        else:
            return whitened


if __name__ == '__main__':
    FeedForwardFuncs.train_from_cfg()

