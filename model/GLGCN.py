import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.layers import Layer
from keras.layers.core import Dropout, Lambda

'''
Model code of GLGCN.
--------
'''


################################################################################################
################################################################################################
# Attention Layers

class TemporalAttention(Layer):
    '''
    compute temporal attention scores
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_timesteps, num_of_timesteps)
    '''

    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.U_1 = self.add_weight(name='U_1',
                                   shape=(num_of_vertices, 1),
                                   initializer='uniform',
                                   trainable=True)
        self.U_2 = self.add_weight(name='U_2',
                                   shape=(num_of_features, num_of_vertices),
                                   initializer='uniform',
                                   trainable=True)
        self.U_3 = self.add_weight(name='U_3',
                                   shape=(num_of_features,),
                                   initializer='uniform',
                                   trainable=True)
        self.b_e = self.add_weight(name='b_e',
                                   shape=(1, num_of_timesteps, num_of_timesteps),
                                   initializer='uniform',
                                   trainable=True)
        self.V_e = self.add_weight(name='V_e',
                                   shape=(num_of_timesteps, num_of_timesteps),
                                   initializer='uniform',
                                   trainable=True)
        super(TemporalAttention, self).build(input_shape)

    def call(self, x):
        _, T, V, F = x.shape

        # shape of lhs is (batch_size, V, T)
        lhs = K.dot(tf.transpose(x, perm=[0, 1, 3, 2]), self.U_1)
        lhs = tf.reshape(lhs, [tf.shape(x)[0], T, F])
        lhs = K.dot(lhs, self.U_2)

        # shape of rhs is (batch_size, T, V)
        rhs = K.dot(self.U_3, tf.transpose(x, perm=[2, 0, 3, 1]))
        rhs = tf.transpose(rhs, perm=[1, 0, 2])

        # shape of product is (batch_size, V, V)
        product = K.batch_dot(lhs, rhs)

        S = tf.transpose(K.dot(self.V_e, tf.transpose(K.sigmoid(product + self.b_e), perm=[1, 2, 0])), perm=[2, 0, 1])

        # normalization
        S = S - K.max(S, axis=1, keepdims=True)
        exp = K.exp(S)
        S_normalized = exp / K.sum(exp, axis=1, keepdims=True)
        return S_normalized

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])



class SpatialAttention(Layer):
    '''
    compute spatial attention scores
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''

    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.W_1 = self.add_weight(name='W_1',
                                   shape=(num_of_timesteps, 1),
                                   initializer='uniform',
                                   trainable=True)
        self.W_2 = self.add_weight(name='W_2',
                                   shape=(num_of_features, num_of_timesteps),
                                   initializer='uniform',
                                   trainable=True)
        self.W_3 = self.add_weight(name='W_3',
                                   shape=(num_of_features,),
                                   initializer='uniform',
                                   trainable=True)
        self.b_s = self.add_weight(name='b_s',
                                   shape=(1, num_of_vertices, num_of_vertices),
                                   initializer='uniform',
                                   trainable=True)
        self.V_s = self.add_weight(name='V_s',
                                   shape=(num_of_vertices, num_of_vertices),
                                   initializer='uniform',
                                   trainable=True)
        super(SpatialAttention, self).build(input_shape)

    def call(self, x):
        _, T, V, F = x.shape

        # shape of lhs is (batch_size, V, T)
        lhs = K.dot(tf.transpose(x, perm=[0, 2, 3, 1]), self.W_1)
        lhs = tf.reshape(lhs, [tf.shape(x)[0], V, F])
        lhs = K.dot(lhs, self.W_2)

        # shape of rhs is (batch_size, T, V)
        rhs = K.dot(self.W_3, tf.transpose(x, perm=[1, 0, 3, 2]))
        rhs = tf.transpose(rhs, perm=[1, 0, 2])

        # shape of product is (batch_size, V, V)
        product = K.batch_dot(lhs, rhs)

        S = tf.transpose(K.dot(self.V_s, tf.transpose(K.sigmoid(product + self.b_s), perm=[1, 2, 0])), perm=[2, 0, 1])

        # normalization
        S = S - K.max(S, axis=1, keepdims=True)
        exp = K.exp(S)
        S_normalized = exp / K.sum(exp, axis=1, keepdims=True)
        return S_normalized

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[2])


################################################################################################
################################################################################################
# Adaptive Graph Learning Layer

def diff_loss(diff, S):
    '''
    compute the 1st loss of L_{graph_learning}
    '''
    if len(S.shape) == 4:
        # batch input
        return K.mean(K.sum(K.sum(diff ** 2, axis=3) * S, axis=(1, 2)))
    else:
        return K.sum(K.sum(diff ** 2, axis=2) * S)


def F_norm_loss(S, Falpha):
    '''
    compute the 2nd loss of L_{graph_learning}
    '''
    if len(S.shape) == 4:
        # batch input
        return Falpha * K.sum(K.mean(S ** 2, axis=0))
    else:
        return Falpha * K.sum(S ** 2)


class Graph_Learn(Layer):
    '''
    Graph structure learning (based on the middle time slice)
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''

    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        self.S = tf.convert_to_tensor([[[0.0]]])  # similar to placeholder
        self.diff = tf.convert_to_tensor([[[[0.0]]]])  # similar to placeholder
        super(Graph_Learn, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.a = self.add_weight(name='a',
                                 shape=(num_of_features, 1),
                                 initializer='uniform',
                                 trainable=True)
        # add loss L_{graph_learning} in the layer
        self.add_loss(F_norm_loss(self.S, self.alpha))
        self.add_loss(diff_loss(self.diff, self.S))
        super(Graph_Learn, self).build(input_shape)

    def call(self, x):
        _, T, V, F = x.shape
        N = tf.shape(x)[0]

        outputs = []
        diff_tmp = 0
        for time_step in range(T):
            # shape: (N,V,F) use the current slice
            xt = x[:, time_step, :, :]
            # shape: (N,V,V)
            diff = tf.transpose(tf.broadcast_to(xt, [V, N, V, F]), perm=[2, 1, 0, 3]) - xt
            # shape: (N,V,V)
            tmpS = K.exp(K.reshape(K.dot(tf.transpose(K.abs(diff), perm=[1, 0, 2, 3]), self.a), [N, V, V]))
            # normalization
            S = tmpS / tf.transpose(tf.broadcast_to(K.sum(tmpS, axis=1), [V, N, V]), perm=[1, 2, 0])

            diff_tmp += K.abs(diff)
            outputs.append(S)

        outputs = tf.transpose(outputs, perm=[1, 0, 2, 3])
        self.S = K.mean(outputs, axis=0)
        self.diff = K.mean(diff_tmp, axis=0) / tf.convert_to_tensor(int(T), tf.float32)
        return outputs

    def compute_output_shape(self, input_shape):
        # shape: (n, num_of_vertices,num_of_vertices, num_of_vertices)
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[2])


################################################################################################
################################################################################################
# GCN layers

class cheb_conv_with_Att_GL(Layer):
    '''
    K-order chebyshev graph convolution with attention after Graph Learn
    --------
    Input:  [x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
             Att (batch_size, num_of_vertices, num_of_vertices),
             S   (batch_size, num_of_vertices, num_of_vertices)]
    Output: (batch_size, num_of_timesteps, num_of_vertices, num_of_filters)
    '''

    def __init__(self, num_of_filters, k, **kwargs):
        self.k = k
        self.num_of_filters = num_of_filters
        super(cheb_conv_with_Att_GL, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        x_shape, Att_shape, S_shape = input_shape
        _, T, V, F = x_shape
        self.Theta = self.add_weight(name='Theta',
                                     shape=(self.k, F, self.num_of_filters),
                                     initializer='uniform',
                                     trainable=True)
        super(cheb_conv_with_Att_GL, self).build(input_shape)

    def call(self, x):
        # Input:  [x, Att, S]
        assert isinstance(x, list)
        assert len(x) == 3, 'Cheb_gcn input error'
        x, Att, S = x
        _, T, V, F = x.shape

        S = K.minimum(S, tf.transpose(S, perm=[0, 1, 3, 2]))  # Ensure symmetry

        # GCN
        outputs = []
        for time_step in range(T):
            # shape of x is (batch_size, V, F)
            graph_signal = x[:, time_step, :, :]
            output = K.zeros(shape=(tf.shape(x)[0], V, self.num_of_filters))

            A = S[:, time_step, :, :]
            # Calculating Chebyshev polynomials (let lambda_max=2)
            D = tf.matrix_diag(K.sum(A, axis=1))
            L = D - A
            L_t = L - [tf.eye(int(V))]
            cheb_polynomials = [tf.eye(int(V)), L_t]
            for i in range(2, self.k):
                cheb_polynomials.append(2 * L_t * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

            for kk in range(self.k):
                T_k = cheb_polynomials[kk]  # shape of T_k is (V, V)
                T_k_with_at = T_k * Att  # shape of T_k_with_at is (batch_size, V, V)
                theta_k = self.Theta[kk]  # shape of theta_k is (F, num_of_filters)

                # shape is (batch_size, V, F)
                rhs = K.batch_dot(tf.transpose(T_k_with_at, perm=[0, 2, 1]), graph_signal)
                output = output + K.dot(rhs, theta_k)
            outputs.append(tf.expand_dims(output, -1))

        return tf.transpose(K.relu(K.concatenate(outputs, axis=-1)), perm=[0, 3, 1, 2])

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        # shape: (n, num_of_timesteps, num_of_vertices, num_of_filters)
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.num_of_filters)


################################################################################################
################################################################################################
# Some operations

def reshape_dot(x):
    # Input:  [x,TAtt]
    x, TAtt = x
    return tf.reshape(
        K.batch_dot(
            tf.reshape(tf.transpose(x, perm=[0, 2, 3, 1]),
                       (tf.shape(x)[0], -1, tf.shape(x)[1])), TAtt),
        [-1, x.shape[1], x.shape[2], x.shape[3]]
    )


def LayerNorm(x):
    # do the layer normalization
    relu_x = K.relu(x)
    ln = tf.contrib.layers.layer_norm(relu_x, begin_norm_axis=3)
    return ln



################################################################################################
################################################################################################
# GLGCN Block

def GLGCN_Block(x, k, num_of_chev_filters, num_of_time_filters, time_conv_strides, time_conv_kernel, GLalpha, i=0):
    '''
    packaged Spatial-temporal convolution Block
    -------
    x: input data;
    k: k-order cheb GCN
    i: block number
    '''

    # temporal attention
    temporal_Att = TemporalAttention()(x)
    x_TAtt = Lambda(reshape_dot, name='reshape_dot' + str(i))([x, temporal_Att])

    # spatial attention
    spatial_Att = SpatialAttention()(x_TAtt)

    S = Graph_Learn(alpha=GLalpha)(x)
    S = Dropout(0.3)(S)
    spatial_gcn_GL = cheb_conv_with_Att_GL(num_of_filters=num_of_chev_filters, k=k)([x, spatial_Att, S])

    # temporal convolution
    time_conv_output_GL = layers.Conv2D(
        filters=num_of_time_filters,
        kernel_size=(time_conv_kernel, 1),
        padding='same',
        strides=(1, time_conv_strides))(spatial_gcn_GL)

    # LayerNorm
    end_output_GL = Lambda(LayerNorm, name='layer_norm' + str(2 * i))(time_conv_output_GL)
    return end_output_GL



################################################################################################
################################################################################################
# GLGCN

def Extract_spatial_temporal(k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                          time_conv_kernel, data_layer_input, num_block, GLalpha
                          ):
    # Input:  (*, num_of_timesteps, num_of_vertices, num_of_features)
    data_layer = data_layer_input

    # GLGCN_Block
    block_out_GL = GLGCN_Block(data_layer, k, num_of_chev_filters, num_of_time_filters,
                               time_conv_strides, time_conv_kernel, GLalpha)
    for i in range(1, num_block):
        block_out_GL = GLGCN_Block(block_out_GL, k, num_of_chev_filters, num_of_time_filters,
                                   time_conv_strides, time_conv_kernel, GLalpha, i)

    block_out = layers.Flatten()(block_out_GL)
    dense_out = layers.Dense(256)(block_out)

    return dense_out
