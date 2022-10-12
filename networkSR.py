#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn import preprocessing
import math

SIZE_IMG = 56
NUM_CHANNELS = 1
NUM_FILTERS = 100
input_kernel_size = 9
output_kernel_size = 5

DICT_DIM =64 # dictionary dimension

def build_cnnlista(inputs_):



    # PATCH EXTRACTION
    newmin = 0.0
    newmax = 1.0
    max = tf.math.reduce_max(inputs_)
    min = tf.math.reduce_min(inputs_)
    tf.cast(max, tf.float32, name=None)
    tf.cast(min, tf.float32, name=None)

    inputs_ = (inputs_ - min)
    inputs_ = inputs_ * (newmax -newmin)
    inputs_ = inputs_ / (max - min)
    inputs_ = inputs_ + newmin

    h = Convolution2D(inputs_,[input_kernel_size, input_kernel_size, NUM_CHANNELS ,NUM_FILTERS],activation=tf.nn.leaky_relu)
    
    # RESHAPE CNN FEATURES
    size_h = tf.shape(h)
    
    print("size of convolutional features:", size_h)


    hh = tf.reshape(h, [-1, size_h[1]*size_h[2], NUM_FILTERS])
    size_hh = tf.shape(hh)
    
    print("size of convolutional features:", size_hh)


    hhh = tf.reshape(hh, [-1, NUM_FILTERS])
    size_hhh = tf.shape(hhh)
   
    print("size of convolutional features:", size_hhh)



    # APPLY LISTA
    lista_features , pow = build_lista(hhh, num_layers = 2, initial_theta = 0.1)
    print("size of LISTA FEATURES:", lista_features)
    print("size of LISTA FEATURES:", lista_features.shape)


    # RECONSTRUCT FROM LISTA FEATURES

    D = np.random.normal(0, 1, (NUM_FILTERS, DICT_DIM))
    D = preprocessing.normalize(D,axis = 1)


    weight3 = np.dot(pow,D)
    weight3_ = tf.Variable(weight3 , dtype=tf.float32, name='weight3')

    hpatch = tf.matmul(lista_features,tf.transpose(weight3_))
    print("size HR after matmul:", hpatch.shape)

    # RESHAPE LISTA FEATURES
    hpatch = tf.reshape(hpatch, [-1, size_h[1], size_h[2], NUM_FILTERS])
    print("size of HR patch FEATURES:", hpatch.shape)


    # PATCH AGGREGATION
    reconstuction = Convolution2D(hpatch,[output_kernel_size, output_kernel_size, NUM_FILTERS, NUM_CHANNELS],activation=tf.nn.leaky_relu)
   

    features = {'input_lista_h': h, 'input_lista_hh': hh,'input_lista_hhh': hhh, 'output_lista':lista_features}

    reconstuction = reconstuction - newmin
    reconstuction = reconstuction * (max -min)
    reconstuction = reconstuction / (newmax - newmin)
    reconstuction = reconstuction + min

    reconstuction=tf.clip_by_value(reconstuction, 0.0  , 255.0, name=None)

    print("size of HR recon:", reconstuction.shape)


    return reconstuction , features


def build_lista(y_, num_layers = 2, initial_theta = 0.1):
    # layer 0
    C = L = 5

    I = np.identity(DICT_DIM)
    W = np.random.normal(0, 1, (NUM_FILTERS, DICT_DIM))
    W = preprocessing.normalize(W,axis = 1)
    WT = W.T


    weight1 = np.dot(C,WT)

    mul = np.dot(WT,W)
    weight2 =I - mul

    W_      = tf.Variable(W, dtype = tf.float32, name = 'W')
    WT_     = tf.Variable(WT, dtype = tf.float32, name = 'W')
    I_      = tf.Variable(I, dtype = tf.float32, name = 'I')
    C_      = tf.Variable(C ,dtype=tf.float32, name='C')
    L_      = tf.Variable(L ,dtype=tf.float32, name='L')

    weight1_ = tf.Variable(weight1,dtype=tf.float32, name='weight1')
    theta0_ = tf.Variable(initial_theta, name='theta0')

    Wy_     = tf.matmul(y_,tf.transpose(weight1_))

    zhat_   = eta(Wy_, theta0_)


    print("size of W_:", W_.shape)
    print("size of Wy_:", Wy_.shape)

    print("size of weight1_:", weight1_.shape)
    

    # layer 1 ... (num_layers-1)
    for t in range(1, num_layers):
        theta_  = tf.Variable(initial_theta, name ='theta_{0}'.format(t) )
        
        w_ = tf.Variable(weight2,dtype=tf.float32, name='weight2')
        wz_     = tf.matmul(zhat_ , tf.transpose(w_))
        zhat_   = eta(wz_ + Wy_, theta_)

        print("size of w_:", w_.shape)
        print("size of wz_:", wz_.shape)
        print("size ofzhat_:", zhat_.shape)

    pow_res = pow(C*L,-1)
    
    print("size ofzhat_:", zhat_.shape)
    return zhat_ , pow_res


def eta(r_, lam_):
    "implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)"
    lam_ = tf.maximum(lam_, 0)
    return tf.sign(r_) * tf.maximum(tf.abs(r_) - lam_, 0)


def Convolution2D(input_tensor,
             kernel_shape,
             strides=(1, 1, 1, 1),
             padding='SAME',
             activation=None,
             scope=''):

        kernel_shape = kernel_shape
        strides = strides
        padding = padding
        activation = activation
        scope = scope


        # build kernel
        
        kernel = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.001), name='kernel')
        # build bias
        kernel_height, kernel_width, num_input_channels, num_output_channels = kernel.get_shape()
        
        bias = tf.Variable(tf.constant(0.1, shape=[num_output_channels]), name='bias')
        # convolution
        conv = tf.nn.conv2d(input_tensor, kernel, strides=strides, padding=padding)
        # activation
        if activation:
            
            return activation(conv + bias )
        return conv + bias


if __name__ == "__main__":
    x = tf.placeholder(tf.float32, shape=[None, 56, 56, NUM_CHANNELS])

    build_cnnlista(x)
