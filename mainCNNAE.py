# created by Alexandros Karampasis
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import problem

import CNNtrain
import CNNtest
import networkSR


TRAIN = 0

scale_rate = 2
filename = "cnnAE_x+"+str(scale_rate)+".npz"

#setup problem
prob = problem.problem()



#build network
decoded_, features_ = networkSR.build_cnnlista(prob.inputs_)


# TRAIN or TEST
if TRAIN:
	
	prob.loadtraindata()
	cost,opt_func =  CNNtrain.setup_training(prob, decoded_)
else:
	prob.loadtestdata()
	cost = CNNtest.setup_testing(prob, decoded_)


print ("Start a new session ...")
sess = tf.Session()

if TRAIN:
    sess.run(tf.global_variables_initializer())
    CNNtrain.load_trainable_vars(sess, "/scales/"+filename)
    CNNtrain.do_training(sess, prob, cost,opt_func,0,"/scales/"+filename)
else:
    filename = 'cnnAE.npz' #file of saved model
    CNNtest.do_testing("/scales/"+filename, sess, prob, cost, decoded_, features_)
