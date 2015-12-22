import theano.tensor as T

from mlutils.gpu import function
from mlutils.parameterset import ParameterSet
from mlmodels.nn import NN


class MLP(NN):

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
            vars.update(self.transfer_func_parameter_shape(lyr, transfer_funcs[lyr], n_units[lyr]))
        self.ps = ParameterSet(**vars)

        # create graph
        v_input = T.fmatrix('v_input')      # v_input[unit, smpl]
        unit_val = [None for _ in range(n_layers)]
        for lyr in range(n_layers):
            if lyr == 0:
                unit_act = v_input
            else:
                unit_act = T.dot(self.weights(lyr - 1, lyr), unit_val[lyr - 1]) + T.shape_padright(self.bias(lyr))
            unit_val[lyr] = self.make_transfer_func(lyr, transfer_funcs[lyr])(unit_act)
        output = unit_val[-1]
        self.f_predict = function([self.ps.flat, v_input], output, name='f_predict')

        # calculate loss
        if loss is not None:
            v_target = T.fmatrix('v_target')    # v_target[unit, smpl]
            fit_smpl_loss = self.fit_loss(loss, transfer_funcs[-1], v_target, output)
            fit_loss = T.mean(fit_smpl_loss)
            loss = fit_loss

            dloss = T.grad(loss, self.ps.flat)
            self.f_loss = function([self.ps.flat, v_input, v_target], loss, name='f_loss')
            self.f_loss_grad = function([self.ps.flat, v_input, v_target], dloss, name='f_loss_grad')


