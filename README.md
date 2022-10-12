# IMAGE-SUPER-RESOLUTION-WITH-DEEP-LEARNING
Image super resolution with deep learning using CNN and LISTA algorithm. Implemented with Tensorflow. 
This project has been implemented during my undergraduate thesis. Special thanks to my teacher L.P. Konti and his assinsant E.Tsiligianni for their help. 

Purpose of this project is to create a deep network model which is capable to produce super resolution images given as input low resolution images. The quality of the producted image is measured by PSNR metric, so i can compare my method with other implementations. The result of this model is very intresting  at optical quality as well as at PSNR values.

![image](https://user-images.githubusercontent.com/17177043/194943425-9b3e5023-260c-4bb1-a51a-8e0abce0e800.png)

This project has been implemented by two sub networks. 

-The first network is a CNN. 
For this network i have implemented a research from Dong, Chao, et al. [1]. In this research they have use a deep convolutional network (CNN) for image super resulution. A graphical approach of this network is given below :

![image](https://user-images.githubusercontent.com/17177043/194945791-ffc6aaf6-e0a7-409e-8920-79902baa3dc3.png)
 
-The second network is a LISTA network.
For this network i have implemented a research from K. Gregor and Y. Le Cun [2]. In this research they have implemented the ISTA algorithm as a network which can compute the sparse representation of the features. A graphical approach of this network is given below :

![image](https://user-images.githubusercontent.com/17177043/194948334-d342065a-5f93-4024-bc99-728837bf7c98.png)

For more informations about these 2 networks please read the correspondingly papers.

------ START OF MY MODEL ------

My model can be divided at three parts according to each different process it compute.

First process: _**Extract features from low resolution image**_

For the features extraction the model uses a convolutional layer H which convolve the input image (low resolution image) with My Filters size of Sx*Sx. 

Second process: _**LISTA network**_

The features from the the first process is used as the input of the LISTA layer. The LISTA network produce the sparse representation α of the features. After the sparse representation is multiplied with a dictionary Dx.   

Third and final process: _**Reconstruct of high resolution image**_

The result of the second process is used as the input of a convlutional layer G. The result of that layer is the reconstructed image (super resolution).

The network of this project can be represanted like : 

![image](https://user-images.githubusercontent.com/17177043/194950173-aa6ff48e-06bc-4110-bc6c-4ed511ad09bb.png)


------ END OF MY MODEL ------

For training  i have used the test91 Dataset.
For testing i have used two datasets. More specific the BSDS100 and Set 14  datasets. 

Link for datasets : https://www.kaggle.com/datasets/msahebi/super-resolution

The model has been tested for thrre different downscale rates.

**RESULTS**

_**X2 SCALE:**_

![image](https://user-images.githubusercontent.com/17177043/195444342-20b9d487-c2a7-409c-9e6a-1f7a15a9f458.png)

_**X3 SCALE:**_

![image](https://user-images.githubusercontent.com/17177043/195444604-fb1cc603-4ae4-4d44-a219-0683ed5dace5.png)

_**X4 SCALE:**_

![image](https://user-images.githubusercontent.com/17177043/195444629-061e2ab5-dbfb-48fa-99b8-b6dd4a6e1c66.png)


PSNR results compare with other researches:

![image](https://user-images.githubusercontent.com/17177043/195448248-3cf16ef7-9c52-4f0c-9964-311ec7424223.png)



HELP INFO :

[1] Dong, Chao, et al. "Image super-resolution using deep convolutional networks." IEEE transactions on pattern analysis and machine intelligence 38.2 (2016): 295-307. https://arxiv.org/pdf/1501.00092.pdf

[2]  Gregor, Karol, and Yann LeCun. "Learning fast approximations of sparse coding." Proceedings of the 27th international conference on international conference on machine learning. 2010. https://icml.cc/Conferences/2010/papers/449.pdf

[3] Timofte, Radu, Vincent De Smet, and Luc Van Gool. "A+: Adjusted anchored neighborhood regression for fast super-resolution." Asian conference on computer vision. Springer, Cham, 2014.

[4] Dong, Chao, et al. "Learning a deep convolutional network for image superresolution." European conference on computer vision. Springer, Cham, 2014.





