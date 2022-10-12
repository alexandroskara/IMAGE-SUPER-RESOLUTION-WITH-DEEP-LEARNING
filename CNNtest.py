#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import copy
import matplotlib.pyplot as plt
import cv2
import psnr
import os

SAVEDIR = "" #whatever path you want
SAVEDIR_figs = "" #whatever path you want
SAVE = 1

def setup_testing(prob, decoded):
    
    cost = tf.losses.mean_squared_error(prob.targets_ ,decoded)
    return cost


def do_testing(filename, sess, prob, cost, decoded_, features_):

    sess.run(tf.global_variables_initializer())
    load_trainable_vars(sess, filename)
    i=0
    for nodataset in range(prob.num_test_datasets):
        sum_cost = 0
        sum_psnr = 0
        for j in range(prob.num_test_data[nodataset]):
            imgs, noisy_imgs = prob.next_test_sample(j, nodataset)

            
            y_lr = noisy_imgs[:,:,0]
            y_lr= y_lr.reshape((y_lr.shape[0], y_lr.shape[1], 1))

            y_hr = imgs[:,:,0]
            y_hr= y_hr.reshape((y_hr.shape[0], y_hr.shape[1], 1))

            input_img = [y_lr]
            target_img = [y_hr]


            batch_cost  = sess.run(cost, feed_dict={prob.inputs_: input_img, prob.targets_: target_img})
            decoded     = sess.run(decoded_, feed_dict={prob.inputs_: input_img})


            estimated_y = decoded[0][:,:,0]

            estimated = copy.deepcopy(noisy_imgs)
            estimated[:,:,0]=estimated_y
            Cb = cv2.split(noisy_imgs)[1]
            Cr = cv2.split(noisy_imgs)[2]


            estimated_image = cv2.merge((estimated,Cb,Cr))
            HRim_arr                = prob.convert_test_ycbcr2rgb(imgs) 
            LRBim_arr               = prob.convert_test_ycbcr2rgb(noisy_imgs)  
            estimated_arr           = prob.convert_test_ycbcr2rgb(estimated_image)

            HRim_arr = HRim_arr.astype(np.uint8)
            LRBim_arr = LRBim_arr.astype(np.uint8)
            estimated_arr = estimated_arr.astype(np.uint8)

            band =  (0, 1, 2) 
            PSNR = psnr.psnr(HRim_arr[:,:,band], estimated_arr[:,:,band])
            if SAVE:
                
                save_images(LRBim_arr, HRim_arr, estimated_arr,i)
            i = i+1
            #plot_testing_results(LRBim_arr, HRim_arr, estimated_arr)
            sum_cost = sum_cost +batch_cost
            sum_psnr = sum_psnr + PSNR


        print("No TESTING DATASET:"+str(nodataset))
        print("Number of testing samples for each dataset: " + str(prob.num_test_data[nodataset]))
        print("Average loss: %f, Average PSNR : %f"% (float(sum_cost/prob.num_test_data[nodataset] ), float(sum_psnr/prob.num_test_data[nodataset] )) )

def check_folder(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
def save_figure(LRBim_arr, HRim_arr, estimated_arr,i):

    foldername= SAVEDIR_figs+"original/"


    check_folder(foldername)
    dpi = 80
    height, width, depth = LRBim_arr.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = 3*(width / float(100)), 2*(height / float(100))
    f = plt.figure(figsize=figsize)
    f.add_subplot(1,3, 1)
    plt.imshow(LRBim_arr)
    plt.title('input')
    plt.axis('off')
    f.add_subplot(1,3, 2)
    plt.imshow(estimated_arr)
    plt.title('estimated ')
    plt.axis('off')
    f.add_subplot(1,3, 3)
    plt.imshow(HRim_arr)
    plt.title('ground truth')
    plt.axis('off')
    #plt.show(block=True)

    plt.savefig(foldername+'plotimage'+str(i)+'.jpg')
    plt.close(f)



def save_images(LRBim_arr, HRim_arr, estimated_arr,i):
    foldername_original = SAVEDIR+"original/"
    foldername_lrb = SAVEDIR+"lrb/"
    foldername_estimated = SAVEDIR+"estimated/"

    check_folder(foldername_original)
    check_folder(foldername_lrb)
    check_folder(foldername_estimated)

    filename_original=foldername_original+'orginal_image'+str(i)+'.jpg'
    cv2.imwrite(filename_original,cv2.cvtColor(HRim_arr, cv2.COLOR_BGR2RGB))

    filename_lrb=foldername_lrb+'lrb_image'+str(i)+'.jpg'
    cv2.imwrite(filename_lrb,cv2.cvtColor(LRBim_arr, cv2.COLOR_BGR2RGB))

    filename_estimated=foldername_estimated+'estimated_image'+str(i)+'.jpg'
    cv2.imwrite(filename_estimated, cv2.cvtColor(estimated_arr, cv2.COLOR_BGR2RGB))



def plot_testing_results(noisy_imgs, imgs, decoded):
    f = plt.figure()
    f.add_subplot(1,3, 1)
    plt.imshow(noisy_imgs)
    plt.title('input')
    f.add_subplot(1,3, 2)
    plt.imshow(imgs)
    plt.title('ground truth')
    f.add_subplot(1,3, 3)
    plt.imshow(decoded)
    plt.title('estimated ')
    plt.show(block=True)


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
                other[k] = d
    except IOError:
        pass
    return other
