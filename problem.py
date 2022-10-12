# created by Alexandros Karampasis
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

import cv2
import os

import random
import utils
import configs as cfgs



# ROUTE PATH
data_path = os.path.dirname(os.getcwd())
print("Looking for data in "+data_path)
# DATA PATH

test_data_path = 'DATASETS/FINAL_TEST'
train_data_path = 'DATASETS'


#------------------#
#train dataset
#------------------#
input_dataHR  = train_data_path + "/test91/original"
input_dataLRB = train_data_path  + "/test91/LRB"


databaseHR = [input_dataHR]
databaseLRB = [input_dataLRB]

#------------------#
# testing dataset
#------------------#
test_data_BSDS100_HR  = test_data_path+"/BSDS100/x2/original"
test_data_BSDS100_LRB = test_data_path+"/BSDS100/x2/LRB"

test_data_Set14_HR  = test_data_path+"/Set14/x2/original"
test_data_Set14_LRB = test_data_path+"/Set14/x2/LRB"

test_data_Set5_HR  = test_data_path+"/Set5/x2/original"
test_data_Set5_LRB = test_data_path+"/Set5/x2/LRB"


BSDS100 = [test_data_BSDS100_HR ,test_data_BSDS100_LRB]
Set14 = [test_data_Set14_HR, test_data_Set14_LRB]
Set5 = [test_data_Set5_HR , test_data_Set5_LRB]

fittrainHR = [input_dataHR, input_dataLRB]

test_datasets = [BSDS100 , Set14 , Set5 ]


class problem():



    def __init__(self):
        self.inputs_ = tf.placeholder(tf.float32, (None, None, None, cfgs.IMAGE_CHANNEL), name='inputs')
        self.targets_ = tf.placeholder(tf.float32, (None, None, None, cfgs.IMAGE_CHANNEL), name='targets')



    def read_train_example(self, fname):
        img_array = cv2.imread(fname)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_array = img_array/255.0
        img_ar = utils.rgb2ycbcr(img_array)
        y, cb, cr = cv2.split(img_ar)
        img_array = y

        return img_array

    def read_test_example_0(self, fname):
        img_array = cv2.imread(fname)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2YCR_CB)
        y, cb, cr = cv2.split(img_array)

        img_arr = np.zeros(img_array.shape)
        img_arr[:, :, 0] = y
        img_arr[:, :, 1] = cb
        img_arr[:, :, 2] = cr
        return img_arr



    def read_test_example(self, fname):
        img_array = cv2.imread(fname)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_array = img_array/255.0
        img_ar = utils.rgb2ycbcr(img_array)
        return img_ar

    def convert_test_ycbcr2rgb(self, img_yuv):
        img_rgb = utils.ycbcr2rgb(img_yuv/255.0) 
        return img_rgb



    def loadtraindata(self):
        training_data_original = []
        training_data_lrb = []

        for i in range(len(databaseHR)):
            for img in os.listdir(databaseHR[i]):
                fname = os.path.join(databaseHR[i],img)
                img_array = self.read_train_example(fname)


                img1 = cv2.flip(img_array, 0)
                img2 = cv2.flip(img_array, -1)
                img3 = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)

                img_ar = np.array(img_array)
                img_ar1 = np.array(img1)
                img_ar2 = np.array(img2)
                img_ar3 = np.array(img3)

                training_data_original.append(img_ar)
                training_data_original.append(img_ar1)
                training_data_original.append(img_ar2)
                training_data_original.append(img_ar3)

        for i in range(len(databaseLRB)):
            for img in os.listdir(databaseLRB[i]):
                fname = os.path.join(databaseLRB[i],img)
                img_array = self.read_train_example(fname)

                img1 = cv2.flip(img_array, 0)
                img2 = cv2.flip(img_array, -1)
                img3 = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)

                img_ar = np.array(img_array)
                img_ar1 = np.array(img1)
                img_ar2 = np.array(img2)
                img_ar3 = np.array(img3)

                training_data_lrb.append(img_ar)
                training_data_lrb.append(img_ar1)
                training_data_lrb.append(img_ar2)
                training_data_lrb.append(img_ar3)

        print("Length training_data_original : " +str(len(training_data_original)))
        print("Length training_data_LRB : " +str(len(training_data_lrb)))
        print("Shape training_data : " +str(training_data_original[0].shape))
        self.data_original = training_data_original
        self.data_LRB = training_data_lrb
        self.num_examples = len(training_data_original)


    def loadtestdata(self):

        
        self.testing_data = []
        self.num_test_datasets = len(test_datasets)
        self.num_test_data = []
        print("Number of testing datasets: " +str(self.num_test_datasets))
        nodataset = 0
        for dataset in test_datasets:
            nodataset = nodataset + 1

            testing_data_original = []
            testing_data_lrb = []
            num_examples_test = 0
            datasetHR = dataset[0]
            datasetLRB = dataset[1]

            for img in os.listdir(datasetHR):
                num_examples_test = num_examples_test + 1

                fname = os.path.join(datasetHR,img)
                testing_data_original.append(self.read_test_example(fname))


            for img in os.listdir(datasetLRB):
                fname = os.path.join(datasetLRB,img)
                testing_data_lrb.append(self.read_test_example(fname))

            self.num_test_data.append(num_examples_test)

            print("Dataset:"+str(nodataset) + ", number of examples: "+str(num_examples_test))
            self.testing_data.append({"HRlist":testing_data_original, "LRBlist":testing_data_lrb})

        print('----------- TESTING DATA LOADED -----------')
        print('Number of datasets loaded: '+str(self.num_test_datasets))
        print('Number of samples in each dataset: '+str(self.num_test_data))
        print("Shape of testing samples: " +str(testing_data_lrb[0].shape))
        print('--------------------------------------------')



    def next_batch(self, batch_size , position):
        original_data_shuffle = [self.data_original[ i] for i in range(position,position+batch_size) ]
        lrb_data_shuffle = [self.data_LRB[ i] for i in range(position,position+batch_size) ]

        or_dat = np.asarray(original_data_shuffle)
        lrb_dat = np.asarray(lrb_data_shuffle)

        HR_imgs = or_dat.reshape((-1,or_dat.shape[1], or_dat.shape[2], cfgs.IMAGE_CHANNEL))
        LR_imgs = lrb_dat.reshape((-1,lrb_dat.shape[1], lrb_dat.shape[2], cfgs.IMAGE_CHANNEL))

        return HR_imgs, LR_imgs

    def next_test_sample(self, noexample, nodataset):
        HRexample = self.testing_data[nodataset]["HRlist"][noexample]
        LRexample = self.testing_data[nodataset]["LRBlist"][noexample]

        return HRexample, LRexample



    def shuffle_data(self):
       
        T = self.num_examples
        index_list = list(range(self.num_examples))
        random.shuffle(index_list)
        self.data_original = self.shuffled_traininglist(index_list, self.data_original)
        self.data_LRB      = self.shuffled_traininglist(index_list, self.data_LRB)


    def shuffled_traininglist(self, index_list, dataset):
        datalist = []
        for index in index_list:
            datalist.append(dataset[index])
        return datalist
