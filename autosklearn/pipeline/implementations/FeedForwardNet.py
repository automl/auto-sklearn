"""
Created on Jul 22, 2015
Modified on Apr 21, 2016

@author: Aaron Klein
@modified: Hector Mendoza
"""
import numpy as np
from sklearn.utils.validation import check_random_state
import theano
import theano.tensor as T
import theano.sparse as S
import lasagne

DEBUG = True


def sharedX(X, dtype=theano.config.floatX, name=None, broadcastable=None):
    if broadcastable:
        return theano.shared(np.asarray(X, dtype=dtype), name=name, broadcastable=broadcastable)
    else:
        return theano.shared(np.asarray(X, dtype=dtype), name=name)


def smorm3s(cost, params, learning_rate=1e-3, eps=1e-16, gather=False):
    updates = []
    optim_params = []
    grads = T.grad(cost, params)

    for p, grad in zip(params, grads):
        mem = sharedX(p.get_value() * 0. + 1., broadcastable=p.broadcastable)
        g = sharedX(p.get_value() * 0., broadcastable=p.broadcastable)
        g2 = sharedX(p.get_value() * 0., broadcastable=p.broadcastable)
        if gather:
            optim_params.append(mem)
            optim_params.append(g)
            optim_params.append(g2)

        r_t = 1. / (mem + 1)
        g_t = (1 - r_t) * g + r_t * grad
        g2_t = (1 - r_t) * g2 + r_t * grad**2
        p_t = p - grad * T.minimum(learning_rate, g_t * g_t / (g2_t + eps)) / \
                  (T.sqrt(g2_t + eps) + eps)
        mem_t = 1 + mem * (1 - g_t * g_t / (g2_t + eps))

        updates.append((g, g_t))
        updates.append((g2, g2_t))
        updates.append((p, p_t))
        updates.append((mem, mem_t))

    return updates


def iterate_minibatches(inputs, targets, batchsize, shuffle=False, random_state=None):
    assert inputs.shape[0] == targets.shape[0],\
           "The number of training points is not the same"
    if shuffle:
        seed = check_random_state(random_state)
        indices = np.arange(inputs.shape[0])
        seed.shuffle(indices)
        # np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


class FeedForwardNet(object):
    def __init__(self, input_shape=(100, 28*28), random_state=None,
                 batch_size=100, num_layers=4, num_units_per_layer=(10, 10, 10),
                 dropout_per_layer=(0.5, 0.5, 0.5), std_per_layer=(0.005, 0.005, 0.005),
                 num_output_units=2, dropout_output=0.5, learning_rate=0.01,
                 lambda2=1e-4, momentum=0.9, beta1=0.9, beta2=0.9,
                 rho=0.95, solver='adam', num_epochs=2,
                 lr_policy='fixed', gamma=0.01, power=1.0, epoch_step=1,
                 activation_per_layer=('relu',)*3, weight_init_per_layer=('henormal',)*3,
                 leakiness_per_layer=(1./3.,)*3, tanh_alpha_per_layer=(2./3.,)*3,
                 tanh_beta_per_layer=(1.7159,)*3,
                 is_sparse=False, is_binary=False, is_regression=False, is_multilabel=False):

        self.random_state = random_state
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.num_units_per_layer = num_units_per_layer
        self.dropout_per_layer = np.asarray(dropout_per_layer, dtype=theano.config.floatX)
        self.num_output_units = num_output_units
        self.dropout_output = T.cast(dropout_output, dtype=theano.config.floatX)
        self.activation_per_layer = activation_per_layer
        self.weight_init_per_layer = weight_init_per_layer
        self.std_per_layer = np.asarray(std_per_layer, dtype=theano.config.floatX)
        self.leakiness_per_layer = np.asarray(leakiness_per_layer, dtype=theano.config.floatX)
        self.tanh_alpha_per_layer = np.asarray(tanh_alpha_per_layer, dtype=theano.config.floatX)
        self.tanh_beta_per_layer = np.asarray(tanh_beta_per_layer, dtype=theano.config.floatX)
        self.momentum = T.cast(momentum, dtype=theano.config.floatX)
        self.learning_rate = np.asarray(learning_rate, dtype=theano.config.floatX)
        self.lambda2 = T.cast(lambda2, dtype=theano.config.floatX)
        self.beta1 = T.cast(beta1, dtype=theano.config.floatX)
        self.beta2 = T.cast(beta2, dtype=theano.config.floatX)
        self.rho = T.cast(rho, dtype=theano.config.floatX)
        self.num_epochs = num_epochs
        self.lr_policy = lr_policy
        self.gamma = np.asarray(gamma, dtype=theano.config.floatX)
        self.power = np.asarray(power, dtype=theano.config.floatX)
        self.epoch_step = np.asarray(epoch_step, dtype=theano.config.floatX)
        self.is_binary = is_binary
        self.is_regression = is_regression
        self.is_multilabel = is_multilabel
        self.is_sparse = is_sparse
        self.solver = solver

        if is_sparse:
            #input_var = S.csr_matrix('inputs', dtype=theano.config.floatX)
            input_var = T.matrix('inputs')
        else:
            input_var = T.matrix('inputs')

        if self.is_binary or self.is_multilabel or self.is_regression:
            target_var = T.matrix('targets')
        else:
            target_var = T.ivector('targets')

        if DEBUG:
            if self.is_binary:
                print("... using binary loss")
            if self.is_multilabel:
                print("... using multilabel prediction")
            if self.is_regression:
                print("... using regression loss")
            print("... building network!")
            print("Input shape:", input_shape)
            print("... with number of epochs:")
            print(num_epochs)

        # Added for reproducibility
        seed = check_random_state(self.random_state)
        lasagne.random.set_rng(seed)

        self.network = lasagne.layers.InputLayer(shape=input_shape,
                                                 input_var=input_var)

        # Define each layer
        for i in range(num_layers - 1):
            init_weight = self._choose_weight_init(i)
            activation_function = self._choose_activation(i)
            self.network = lasagne.layers.DenseLayer(
                 lasagne.layers.dropout(self.network,
                                        p=self.dropout_per_layer[i]),
                 num_units=self.num_units_per_layer[i],
                 W=init_weight,
                 b=lasagne.init.Constant(val=0.0),
                 nonlinearity=activation_function)

        # Define output layer and nonlinearity of last layer
        if self.is_regression:
            output_activation = lasagne.nonlinearities.linear
        elif self.is_binary or self.is_multilabel:
            output_activation = lasagne.nonlinearities.sigmoid
        else:
            output_activation = lasagne.nonlinearities.softmax

        self.network = lasagne.layers.DenseLayer(
                 lasagne.layers.dropout(self.network,
                                        p=self.dropout_output),
                 num_units=self.num_output_units,
                 W=lasagne.init.GlorotNormal(),
                 b=lasagne.init.Constant(),
                 nonlinearity=output_activation)

        prediction = lasagne.layers.get_output(self.network)

        if self.is_regression:
            loss_function = lasagne.objectives.squared_error
        elif self.is_binary or self.is_multilabel:
            loss_function = lasagne.objectives.binary_crossentropy
        else:
            loss_function = lasagne.objectives.categorical_crossentropy

        loss = loss_function(prediction, target_var)

        # Aggregate loss mean function with l2
        # Regularization on all layers' params
        if self.is_binary or self.is_multilabel:
            #loss = T.sum(loss, dtype=theano.config.floatX)
            loss = T.mean(loss, dtype=theano.config.floatX)
        else:
            loss = T.mean(loss, dtype=theano.config.floatX)
        l2_penalty = self.lambda2 * lasagne.regularization.regularize_network_params(
            self.network, lasagne.regularization.l2)
        loss += l2_penalty
        params = lasagne.layers.get_all_params(self.network, trainable=True)

        # Create the symbolic scalar lr for loss & updates function
        lr_scalar = T.scalar('lr', dtype=theano.config.floatX)

        if solver == "nesterov":
            updates = lasagne.updates.nesterov_momentum(loss, params,
                                                        learning_rate=lr_scalar,
                                                        momentum=self.momentum)
        elif solver == "adam":
            updates = lasagne.updates.adam(loss, params,
                                           learning_rate=lr_scalar,
                                           beta1=self.beta1, beta2=self.beta2)
        elif solver == "adadelta":
            updates = lasagne.updates.adadelta(loss, params,
                                               learning_rate=lr_scalar,
                                               rho=self.rho)
        elif solver == "adagrad":
            updates = lasagne.updates.adagrad(loss, params,
                                              learning_rate=lr_scalar)
        elif solver == "sgd":
            updates = lasagne.updates.sgd(loss, params,
                                          learning_rate=lr_scalar)
        elif solver == "momentum":
            updates = lasagne.updates.momentum(loss, params,
                                               learning_rate=lr_scalar,
                                               momentum=self.momentum)
        elif solver == "smorm3s":
            updates = smorm3s(loss, params,
                              learning_rate=lr_scalar)
        else:
            updates = lasagne.updates.sgd(loss, params,
                                          learning_rate=lr_scalar)

        # Validation was removed, as auto-sklearn handles that, if this net
        # is to be used independently, validation accuracy has to be included
        if DEBUG:
            print("... compiling theano functions")
        self.train_fn = theano.function([input_var, target_var, lr_scalar],
                                        loss,
                                        updates=updates,
                                        allow_input_downcast=True,
                                        profile=False,
                                        on_unused_input='warn',
                                        name='train_fn')
        if DEBUG:
            print('... compiling update function')
        self.update_function = self._policy_function()

    def _policy_function(self):
        epoch, gm, powr, step = T.scalars('epoch', 'gm', 'powr', 'step')
        if self.lr_policy == 'inv':
            decay = T.power(1.0+gm*epoch, -powr)
        elif self.lr_policy == 'exp':
            decay = gm ** epoch
        elif self.lr_policy == 'step':
            decay = T.switch(T.eq(T.mod_check(epoch, step), 0.0),
                             T.power(gm, T.floor_div(epoch, step)),
                             1.0)
        else:
            decay = T.constant(1.0, name='fixed', dtype=theano.config.floatX)

        return theano.function([gm, epoch, powr, step],
                               decay,
                               allow_input_downcast=True,
                               on_unused_input='ignore',
                               name='update_fn')

    def fit(self, X, y):
        if self.batch_size > X.shape[0]:
            self.batch_size = X.shape[0]
            print('One update per epoch batch size')

        if self.is_sparse:
            X = X.astype(np.float32)
        else:
            try:
                X = np.asarray(X, dtype=theano.config.floatX)
                y = np.asarray(y, dtype=theano.config.floatX)
            except Exception as E:
                print('Fit casting error: %s' % E)

        for epoch in range(self.num_epochs):
            train_err = 0
            train_batches = 0
            for inputs, targets in iterate_minibatches(X, y, self.batch_size, shuffle=True,
                                                       random_state=self.random_state):
                if self.is_sparse:
                    inputs = inputs.toarray()
                train_err += self.train_fn(inputs, targets, self.learning_rate)
                train_batches += 1
            decay = self.update_function(self.gamma, epoch+1.0,
                                         self.power, self.epoch_step)
            self.learning_rate *= decay
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        return self

    def predict(self, X, is_sparse=False):
        predictions = self.predict_proba(X, is_sparse)
        if self.is_multilabel:
            return np.round(predictions)
        elif self.is_regression:
            return predictions
        else:
            return np.argmax(predictions, axis=1)

    def predict_proba(self, X, is_sparse=False):
        if is_sparse:
            #X = X.astype(np.float32)
            #X = S.as_sparse_or_tensor_variable(X)
            X = X.toarray()
        else:
            try:
                X = np.asarray(X, dtype=theano.config.floatX)
            except Exception as E:
                print('Prediction casting error: %s' % E)

        predictions = lasagne.layers.get_output(self.network,
                                                X, deterministic=True).eval()
        if self.is_binary:
            return np.append(1.0 - predictions, predictions, axis=1)
        else:
            return predictions

    def _choose_activation(self, index=0, output=False):
        if output:
            nl = getattr(self, 'output_activations', None)
        else:
            nl = getattr(self, 'activation_functions', None)

        activation = self.activation_per_layer[index]
        layer_activation = nl.get(activation)
        if activation == 'scaledTanh':
            layer_activation = layer_activation(scale_in=self.tanh_alpha_per_layer[index],
                                                scale_out=self.tanh_beta_per_layer[index])
        elif activation == 'leaky':
            layer_activation = layer_activation(leakiness=self.leakiness_per_layer[index])

        return layer_activation

    def _choose_weight_init(self, index=0, output=False):
        wi = getattr(self, 'weight_initializations', None)
        initialization = self.weight_init_per_layer[index]
        weight_init = wi.get(initialization)
        if initialization == 'normal':
            weight_init = weight_init(std=self.std_per_layer[index])
        else:
            weight_init = weight_init()

        return weight_init

    activation_functions = {
        'relu': lasagne.nonlinearities.rectify,
        'leaky': lasagne.nonlinearities.LeakyRectify,
        'very_leaky': lasagne.nonlinearities.very_leaky_rectify,
        'elu': lasagne.nonlinearities.elu,
        'linear': lasagne.nonlinearities.linear,
        'scaledTanh': lasagne.nonlinearities.ScaledTanH,
        'sigmoid': lasagne.nonlinearities.sigmoid,
        'tahn': lasagne.nonlinearities.tanh,
    }
    output_activations = {
        'softmax': lasagne.nonlinearities.softmax,
        'softplus': lasagne.nonlinearities.softplus,
        'sigmoid': lasagne.nonlinearities.sigmoid,
        'tahn': lasagne.nonlinearities.tanh,
    }
    weight_initializations = {
        'constant': lasagne.init.Constant,
        'normal': lasagne.init.Normal,
        'uniform': lasagne.init.Uniform,
        'glorot_normal': lasagne.init.GlorotNormal,
        'glorot_uniform': lasagne.init.GlorotUniform,
        'he_normal': lasagne.init.HeNormal,
        'he_uniform': lasagne.init.HeUniform,
        'ortogonal': lasagne.init.Orthogonal,
        'sparse': lasagne.init.Sparse
    }

