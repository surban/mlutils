import theano.tensor as T
from logging import warning
from mlutils.gpu import function
from mlutils.parameterset import ParameterSet


class MLP(object):

    def __init__(self, loss, n_units, transfer_funcs):
        n_layers = len(n_units)

        print "===== MLP ========="
        print "Number of layers:  ", n_layers
        print "Loss:              ", loss
        print "Number of units:   ", n_units
        print "Transfer function: ", transfer_funcs
        print "==================="

        # create ParameterSet
        vars = {}
        for lyr in range(n_layers):
            if lyr != 0:
                vars["weights_%d_to_%d" % (lyr - 1, lyr)] = (n_units[lyr], n_units[lyr - 1])
                vars["bias_%d" % lyr] = (n_units[lyr],)
            if transfer_funcs[lyr] in ['abel', 'abel_tanh', 'sigm_abel', 'abel_sigm', 'relu_abel']:
                vars["n_%d" % lyr] = (n_units[lyr],)
            if transfer_funcs[lyr] in ['abel_sigm_abel']:
                vars["n_in_%d" % lyr] = (n_units[lyr],)
                vars["n_out_%d" % lyr] = (n_units[lyr],)
        self.ps = ParameterSet(**vars)

        def weights(frm, to):
            return self.ps.sym("weights_%d_to_%d" % (frm, to))
        def bias(lyr):
            return self.ps.sym("bias_%d" % lyr)
        def abel_n(lyr):
            return self.ps.sym("n_%d" % lyr)
        def make_transfer_func(lyr):
            tf = transfer_funcs[lyr]
            if tf == 'tanh':
                return T.tanh
            elif tf == 'sigm':
                return T.nnet.sigmoid
            elif tf == 'softmax':
                return T.nnet.softmax
            elif tf == 'linear':
                return lambda x: x
            elif tf == 'relu':
                return T.nnet.relu
            elif tf == 'exp':
                return T.exp
            elif tf == 'log':
                return lambda x: T.log(T.maximum(0.0001, x))
            elif tf == 'sin':
                return T.sin
            elif tf == 'psi':
                from addiplication.nnet.abelpsi_f import abel_psi_f
                return abel_psi_f
            elif tf == 'psiinv':
                from addiplication.nnet.abelpsi_f import abel_psi_inv_f
                return abel_psi_inv_f
            elif tf == 'abel':
                from addiplication.nnet.abelpsi_f import abel_fracexpn_f
                return lambda x: abel_fracexpn_f(x, T.shape_padright(abel_n(lyr)))
            elif tf == 'relu_abel':
                from addiplication.nnet.abelpsi_f import abel_fracexpn_f
                return lambda x: abel_fracexpn_f(T.nnet.relu(x), T.shape_padright(abel_n(lyr)))
            elif tf == 'sigm_abel':
                from addiplication.nnet.abelpsi_f import abel_fracexpn_f
                return lambda x: abel_fracexpn_f(T.nnet.sigmoid(x), T.shape_padright(abel_n(lyr)))
            elif tf == 'abel_sigm':
                from addiplication.nnet.abelpsi_f import abel_fracexpn_f
                return lambda x: T.nnet.sigmoid(abel_fracexpn_f(x, T.shape_padright(abel_n(lyr))))
            elif tf == 'abel_tanh':
                from addiplication.nnet.abelpsi_f import abel_fracexpn_f
                return lambda x: T.tanh(abel_fracexpn_f(x, T.shape_padright(abel_n(lyr))))
            elif tf == 'tanh_abel':
                from addiplication.nnet.abelpsi_f import abel_fracexpn_f
                return lambda x: abel_fracexpn_f(T.tanh(x), T.shape_padright(abel_n(lyr)))
            elif tf == 'abel_sigm_abel':
                from addiplication.nnet.abelpsi_f import abel_fracexpn_f
                def abel_sigm_abel_tf(x):
                    abel1 = abel_fracexpn_f(x, T.shape_padright(self.ps.sym("n_in_%d" % lyr)))
                    sigm = T.nnet.sigmoid(abel1) + 0.05
                    abel2 = abel_fracexpn_f(sigm, T.shape_padright(self.ps.sym("n_out_%d" % lyr)))
                    return abel2
                return lambda x: abel_sigm_abel_tf(x)
            else:
                raise ValueError("unknown transfer function %s for layer %d" % (tf, lyr))

        # create graph
        v_input = T.fmatrix('v_input')      # v_input[unit, smpl]
        unit_val = [None for _ in range(n_layers)]
        for lyr in range(n_layers):
            if lyr == 0:
                unit_act = v_input
            else:
                unit_act = T.dot(weights(lyr - 1, lyr), unit_val[lyr - 1]) + T.shape_padright(bias(lyr))
            unit_val[lyr] = make_transfer_func(lyr)(unit_act)
        output = unit_val[-1]
        self.f_predict = function([self.ps.flat, v_input], output, name='f_predict')

        # calculate loss
        if loss is not None:
            v_target = T.fmatrix('v_target')    # v_target[unit, smpl]
            if loss == 'cross_entropy':
                if transfer_funcs[-1] not in ['sigm', 'softmax']:
                    warning("using cross-entropy loss without softmax or sigmoid transfer function")
                smpl_loss = -T.sum(T.xlogx.xlogy0(v_target, T.maximum(0.001, output)), axis=0)
                if transfer_funcs[-1] == 'sigm':
                    smpl_loss += -T.sum(T.xlogx.xlogy0(1 - v_target, T.maximum(0.001, 1 - output)), axis=0)
            elif loss == 'l2':
                smpl_loss = T.sum((v_target - output)**2, axis=0)
            else:
                raise ValueError("unknown loss %s" % loss)
            fit_loss = T.mean(smpl_loss)
            loss = fit_loss

            dloss = T.grad(loss, self.ps.flat)
            self.f_loss = function([self.ps.flat, v_input, v_target], loss, name='f_loss')
            self.f_loss_grad = function([self.ps.flat, v_input, v_target], dloss, name='f_loss_grad')





