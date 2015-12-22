import theano.tensor as T
from theano.scan_module.scan import scan
from mlutils.misc import random_matrix_with_spectral_radius, print_node
from mlutils.modelfuncs import ModelFuncs
from mlutils.parameterset import ParameterSet
from mlutils.gpu import function
from mlmodels.nn import NN


class RNN(NN):

    def __init__(self, loss, n_units, transfer_funcs, gradient_steps=-1, n_feedback_steps=0):
        assert len(n_units) == 3, "currently RNN must consists of one input, one hidden and one output layer"
        
        n_inputs, n_hiddens, n_outputs = n_units
        input_tf, hidden_tf, output_tf = transfer_funcs 
        
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        self.input_tf = input_tf
        self.hidden_tf = hidden_tf
        self.output_tf = output_tf

        print "====== RNN ========"
        print "Loss:              ", loss
        print "Number of units:   ", n_units
        print "Transfer function: ", transfer_funcs
        print "==================="

        # create ParameterSet
        pars = dict(inp_hid=(n_hiddens, n_inputs),
                    hid_hid=(n_hiddens, n_hiddens),
                    hid_bias=(n_hiddens,),
                    hid_out=(n_outputs, n_hiddens),
                    out_bias=(n_outputs,))
        pars.update(self.transfer_func_parameter_shape('inp', input_tf, n_hiddens))                   
        pars.update(self.transfer_func_parameter_shape('hid', hidden_tf, n_hiddens))
        pars.update(self.transfer_func_parameter_shape('out', output_tf, n_outputs))
        self.ps = ParameterSet(**pars)

        ###########################################################################
        # RNN recursion
        ###########################################################################

        def recursion(inp_act, prv_hid):
            # input 
            inp = self.make_transfer_func('inp', input_tf)(inp_act)
        
            # hiddens given inputs and previous hiddens
            in_act = T.dot(self.ps.inp_hid, inp)
            prv_hid_act = T.dot(self.ps.hid_hid, prv_hid)
            hid_bias_bc = T.shape_padright(self.ps.hid_bias)
            hid_act = prv_hid_act + in_act + hid_bias_bc
            hid = self.make_transfer_func('hid', hidden_tf)(hid_act)

            # outputs given hiddens
            out_bias_bc = T.shape_padright(self.ps.out_bias)
            out_act = T.dot(self.ps.hid_out, hid) + out_bias_bc
            out = self.make_transfer_func('out', output_tf)(out_act)

            return out, hid

        ###########################################################################
        # prediction and loss
        ###########################################################################
        v_valid = T.fmatrix('v_valid')                  # v_valid[step, smpl]
        v_output = T.ftensor3('v_output')               # v_output[channel, step, smpl]
        v_input = T.ftensor3('v_input')                 # v_input[channel, step, smpl]
        n_samples = v_input.shape[2]

        # calculate predictions
        hid_init = T.zeros((n_hiddens, n_samples))
        (pred_scan, _), _ = scan(recursion,
                                 sequences=[{'input': v_input.dimshuffle(1, 0, 2), 'taps': [0]}],
                                 outputs_info=[None,
                                               {'initial': hid_init, 'taps': [-1]}],
                                 truncate_gradient=gradient_steps)
        pred = pred_scan.dimshuffle(1, 0, 2)            # pred[channel, step, smpl]

        # calculate loss
        # step_loss[step, smpl]
        step_loss = self.fit_loss(loss, output_tf, v_output, pred)
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

