# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

import matplotlib.pyplot as plt


epochs = 5000
batch_size = 128
learning_rate = 0.0001


def setup_training(prob, decoded):
    print("set up training ...")
    
    cost = tf.reduce_mean(tf.square(prob.targets_ - decoded))
    optimizer = tf.train.AdamOptimizer(1e-4)
    gradients, variables = zip(*optimizer.compute_gradients(cost))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    opt_func = optimizer.apply_gradients(zip(gradients, variables))
    return cost ,opt_func


def do_training(sess, prob, cost ,opt_func,first_time,filename):


    num_batches = int(prob.num_examples/batch_size)
    print("num_batches", num_batches)

    if first_time == 1 :
        load_trainable_vars(sess, 'AE.npz')

    train_cost = []
    print("Start training: num epochs = ", epochs)
    for e in range(epochs):

        file1 = open("MyFile.txt","a")
     
        training = 0
       
        start_position = 0
        for ii in range(num_batches):


            imgs, noisy_imgs = prob.next_batch(batch_size , start_position)

            start_position = start_position + batch_size

            batch_cost, _ = sess.run([cost, opt_func], feed_dict={prob.inputs_: noisy_imgs, prob.targets_: imgs })
            train_cost.append(batch_cost)
            training = training + batch_cost


        prob.shuffle_data()
        print("Epoch: {}/{}: ".format(e, epochs), "Training loss: {:.7f}".format(training/num_batches))
        file1.write("Epoch: {} , train_loss : {} \n".format(e+1,(training/num_batches)))
        save_trainable_vars(sess,filename)

        file1.close()
    plt.plot(train_cost)
    plt.title('training loss')
    plt.savefig('train_loss.png')
    plt.show()




def save_trainable_vars(sess,filename,**kwargs):
    """save a .npz archive in `filename`  with
    the current value of each variable in tf.trainable_variables()
    plus any keyword numpy arrays.
    """
    print("Saving trainable variables to %s"%filename)
    save={}
    for v in tf.trainable_variables():
        save[str(v.name)] = sess.run(v)
    save.update(kwargs)
    np.savez(filename,**save)




def load_trainable_vars(sess,filename):
    """load a .npz archive and assign the value of each loaded
    ndarray to the trainable variable whose name matches the
    archive key.  Any elements in the archive that do not have
    a corresponding trainable variable will be returned in a dict.
    """
    print("Loading learned variables from %s"%filename)
    other={}
    try:
        tv=dict([ (str(v.name),v) for v in tf.trainable_variables() ])
        for k,d in np.load(filename).items():
            if k in tv:
                print('restoring ' + k)
                sess.run(tf.assign( tv[k], d) )
            else:
                print("Cannot Load")
                other[k] = d
    except IOError:
        pass
    return other
