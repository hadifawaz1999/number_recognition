import cv2
import matplotlib.pyplot as plt
import numpy as np
import keras
from math import pow

img_path="/home/hadi/vstudioProjects/digit_recog/test_num_recg.png"
img=cv2.imread(img_path)
img=cv2.imwrite("test_num_recg.png",img[:,:,1])
img=cv2.imread("test_num_recg.png",cv2.IMREAD_UNCHANGED)
img=np.asarray(img)
img=np.split(img,2,axis=1)
img=np.asarray(img,dtype=np.float64)
img.shape=(-1,28,28,1)
img/=255
my_model=keras.models.load_model("/home/hadi/vstudioProjects/digit_recog/my_model.hdf5")
pred=my_model.predict(img,verbose=False)
pred=np.argmax(pred,axis=1)
x=0
l=int(pred.shape[0])
for i in range(l):
    x+=pred[i]*pow(10,l-i-1)
print(int(x))