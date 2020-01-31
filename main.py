import cv2
import matplotlib.pyplot as plt
import numpy as np
import keras
from math import pow
import random

img_path="test_num_recg.png"
img=cv2.imread(img_path)
img=cv2.imwrite("test_num_recg.png",img[:,:,1])
img=cv2.imread("test_num_recg.png",cv2.IMREAD_UNCHANGED)
img=np.asarray(img)

indices=[]
digit_before=False
num_digits=0
for i in range(img.shape[1]):
    if np.sum(img[:,i])==0 and digit_before:
        num_digits+=1
        indices.append(i+1)
        i+=1
        digit_before=False
    elif np.sum(img[:,i])!=0:
        digit_before=True

img=np.split(img,indices,axis=1)
for i in range(num_digits):
    plt.imsave('new_data/img'+str(i)+'.png',img[i],cmap='gray')
digits=[]
data=[]
for i in range(num_digits):
    img=cv2.imread('new_data/img'+str(i)+'.png')
    img=img[:,:,1]
    if img.shape[1]<28:
        print("test1")
        size_to_add=28-img.shape[1]
        while size_to_add!=0:
            zeros=np.zeros(shape=(28,1))
            if random.random()>0.5:
                img=np.concatenate((img,zeros),axis=1)
            else:
                img=np.concatenate((zeros,img),axis=1)
            size_to_add-=1
        # zeros=np.zeros(shape=(28,size_to_add))
        # img=np.concatenate((img,zeros),axis=1)
    elif img.shape[1]>28:
        print("test2")
        new_shape=(28,28)
        img=cv2.resize(img,new_shape)
    # plt.imshow(img,cmap='gray')
    # plt.show()
    
    """
    new_shape=(28,28)
    img=cv2.resize(img,new_shape)
    """

    plt.imsave('new_data/img'+str(i)+'.png',img,cmap='gray')
    img=np.asarray(img,dtype=np.float64)
    img.shape=(28,28,1)
    data.append(img)
data=np.asarray(data,dtype=np.float64)
data/=255
my_model=keras.models.load_model("my_model.hdf5")
pred=my_model.predict(data,verbose=False)
pred=np.argmax(pred,axis=1)
number=0
for i in range(pred.shape[0]):
    number+=pred[i]*pow(10,int(pred.shape[0])-i-1)
print(int(number))