import theano.tensor as T
from logging import warning


class NN(object):

    def weights(self, frm, to):
        return self.ps.sym("weights_%s_to_%s" % (str(frm), str(to)))
        
    def bias(self, lyr):
        return self.ps.sym("bias_%s" % str(lyr))
        
    def abel_n(self, lyr):
        return self.ps.sym("n_%s" % str(lyr))

    def abel_in_n(self, lyr):
        return self.ps.sym("n_in_%s" % str(lyr))

    def abel_out_n(self, lyr):
        return self.ps.sym("n_out_%s" % str(lyr))
        
    def make_transfer_func(self, lyr, tf):
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
            return lambda x: abel_fracexpn_f(x, T.shape_padright(self.abel_n(lyr)))
        elif tf == 'relu_abel':
            from addiplication.nnet.abelpsi_f import abel_fracexpn_f
            return lambda x: abel_fracexpn_f(T.nnet.relu(x), T.shape_padright(self.abel_n(lyr)))
        elif tf == 'sigm_abel':
            from addiplication.nnet.abelpsi_f import abel_fracexpn_f
            return lambda x: abel_fracexpn_f(T.nnet.sigmoid(x), T.shape_padright(self.abel_n(lyr)))
        elif tf == 'abel_sigm':
            from addiplication.nnet.abelpsi_f import abel_fracexpn_f
            return lambda x: T.nnet.sigmoid(abel_fracexpn_f(x, T.shape_padright(self.abel_n(lyr))))
        elif tf == 'abel_tanh':
            from addiplication.nnet.abelpsi_f import abel_fracexpn_f
            return lambda x: T.tanh(abel_fracexpn_f(x, T.shape_padright(self.abel_n(lyr))))
        elif tf == 'tanh_abel':
            from addiplication.nnet.abelpsi_f import abel_fracexpn_f
            return lambda x: abel_fracexpn_f(T.tanh(x), T.shape_padright(self.abel_n(lyr)))
        elif tf == 'abel_sigm_abel':
            from addiplication.nnet.abelpsi_f import abel_fracexpn_f
            def abel_sigm_abel_tf(x):
                abel1 = abel_fracexpn_f(x, T.shape_padright(self.abel_in_n(lyr)))
                sigm = T.nnet.sigmoid(abel1) + 0.05
                abel2 = abel_fracexpn_f(sigm, T.shape_padright(self.abel_out_n(lyr)))
                return abel2
            return abel_sigm_abel_tf
        else:
            raise ValueError("unknown transfer function %s for layer %s" % (tf, str(lyr)))

    def transfer_func_parameter_shape(self, lyr, tf, n_units):    
        if tf in ['abel', 'abel_tanh', 'sigm_abel', 'abel_sigm', 'relu_abel']:
            return {"n_%s" % str(lyr): (n_units,)}
        elif tf in ['abel_sigm_abel']:
            return {"n_in_%s" % str(lyr): (n_units,),
                    "n_out_%s" % str(lyr): (n_units,)}
        else:
            return {}
            
    def fit_loss(self, loss, tf, target, output):
        min_prob = 1e-3
        if loss == 'cross_entropy':
            if tf != 'softmax':
                warning("using cross-entropy loss without softmax transfer function")
            smpl_loss = -T.sum(T.xlogx.xlogy0(v_target, T.maximum(min_prob, output)), axis=0)
        elif loss == 'cross_entropy_binary':
            if tf != 'sigmoid':
                warning("using binary cross-entropy loss without sigmoid transfer function")       
            smpl_loss = -T.sum(T.xlogx.xlogy0(v_target, T.maximum(min_prob, output)), axis=0)
            smpl_loss += -T.sum(T.xlogx.xlogy0(1 - v_target, T.maximum(min_prob, 1.0 - output)), axis=0)
        elif loss == 'l2':
            smpl_loss = T.sum((target - output)**2, axis=0)
        else:
            raise ValueError("unknown loss %s" % loss)
        return smpl_loss
    
    
            
                    
