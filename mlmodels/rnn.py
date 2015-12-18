import matplotlib.pyplot as plt
import theano.tensor as T
import numpy as np
from os.path import join
from theano.scan_module.scan import scan

from addiplication.nnet.abelpsi_f import abel_fracexpn_f
from midi.utils import midiwrite
from mlutils.config import load_cfg
from mlutils.dataset import Dataset
from mlutils.gpu import function, gather, post
from mlutils.misc import random_matrix_with_spectral_radius, print_node
from mlutils.modelfuncs import ModelFuncs
from mlutils.parameterset import ParameterSet
from mlutils.preprocess import for_step_data, pca_white, pca_white_inverse


class Rnn(object):

    def __init__(self, cfg, n_inputs, n_outputs, n_feedback_steps=0):
        self.cfg = cfg
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        print "Number of inputs:  ", n_inputs
        print "Number of hiddens: ", cfg.n_hiddens
        print "Number of outputs: ", n_outputs
        print "Transfer function: ", cfg.transfer_func

        # create ParameterSet
        pars = dict(in_hid=(cfg.n_hiddens, n_inputs),
                    hid_hid=(cfg.n_hiddens, cfg.n_hiddens),
                    hid_bias=(cfg.n_hiddens,),
                    hid_out=(n_outputs, cfg.n_hiddens),
                    out_bias=(n_outputs,))
        if cfg.transfer_func == 'abel':
            pars['hid_n'] = (cfg.n_hiddens,)
            pars['partitions'] = {'n': ['hid_n']}
        ps = ParameterSet(**pars)
        self.ps = ps

        def recursion(inp, prv_hid):
            # hiddens given inputs and previous hiddens
            prv_hid_act = T.dot(ps.hid_hid, prv_hid)
            in_act = T.dot(ps.in_hid, inp)
            hid_bias_bc = T.shape_padright(ps.hid_bias)
            hid_act = prv_hid_act + in_act + hid_bias_bc
            if cfg.transfer_func == 'tanh':
                hid = T.tanh(hid_act)
            elif cfg.transfer_func == 'linear':
                hid = hid_act
            elif cfg.transfer_func == 'relu':
                hid = T.nnet.relu(hid_act)
            elif cfg.transfer_func == 'abel':
                # hid_act = T.printing.Print("hid_act")(hid_act)
                hid_relu = T.nnet.relu(hid_act)
                # hid_relu = T.printing.Print("hid_relu")(hid_relu)
                hid = abel_fracexpn_f(hid_relu, T.shape_padright(self.ps.hid_n))
                # hid = T.printing.Print("hid")(hid)
            else:
                assert False

            # outputs given hiddens
            out_act = T.dot(self.ps.hid_out, hid)
            out_bias_bc = T.shape_padright(self.ps.out_bias)
            if cfg.loss == 'cross_entropy':
                out = T.nnet.sigmoid(out_act + out_bias_bc)
            elif cfg.loss == 'l2':
                out = out_act + out_bias_bc
            else:
                assert False

            return out, hid

        ###########################################################################
        # prediction and loss
        ###########################################################################
        v_valid = T.fmatrix('v_valid')                  # v_valid[step, smpl]
        v_output = T.ftensor3('v_output')               # v_output[channel, step, smpl]
        v_input = T.ftensor3('v_input')                 # v_input[channel, step, smpl]
        n_samples = v_input.shape[2]

        # calculate predictions
        hid_init = T.zeros((cfg.n_hiddens, n_samples))
        (pred_scan, _), _ = scan(recursion,
                                 sequences=[{'input': v_input.dimshuffle(1, 0, 2), 'taps': [0]}],
                                 outputs_info=[None,
                                               {'initial': hid_init, 'taps': [-1]}],
                                 truncate_gradient=cfg.gradient_steps)
        pred = pred_scan.dimshuffle(1, 0, 2)            # pred[channel, step, smpl]

        # calculate loss
        # step_loss[step, smpl]
        if cfg.loss == 'cross_entropy':
            print "Using cross entropy loss"
            step_loss = -T.sum(T.xlogx.xlogy0(v_output, T.maximum(0.001, pred)), axis=0)
            step_loss += -T.sum(T.xlogx.xlogy0(1 - v_output, T.maximum(0.001, 1 - pred)), axis=0)
        elif cfg.loss == 'l2':
            print "Using L2^2 loss"
            step_loss = T.sum((v_output - pred)**2, axis=0)
        else:
            assert False
        loss = T.sum(v_valid * step_loss) / T.sum(v_valid)
        loss_grad = T.grad(loss, self.ps.flat)

        # define functions
        self.f_predict = function([self.ps.flat, v_input], pred, name='f_predict')
        self.f_loss = function([self.ps.flat, v_valid, v_input, v_output], loss,
                               name='f_loss', on_unused_input='warn')
        self.f_loss_grad = function([self.ps.flat, v_valid, v_input, v_output], loss_grad,
                                    name='f_loss_grad', on_unused_input='warn')

        ###########################################################################
        # data generation using feedback
        ###########################################################################
        if n_feedback_steps > 0:
            v_data_init = T.fmatrix('v_data_init')      # v_data_init[channel, smpl]
            v_hid_init = T.fmatrix('v_hid_init')        # v_hid_init[unit, smpl]

            (feedback_out_scan, _), _ = scan(recursion,
                                             outputs_info=[{'initial': v_data_init, 'taps': [-1]},
                                                           {'initial': v_hid_init, 'taps': [-1]}],
                                             n_steps=n_feedback_steps)
            feedback_out = feedback_out_scan.dimshuffle(1, 0, 2)  # feedback_out[channel, step, smpl]
            self.f_feedback_generate = function([self.ps.flat, v_data_init, v_hid_init], feedback_out,
                                                name='f_feedback_generate')

