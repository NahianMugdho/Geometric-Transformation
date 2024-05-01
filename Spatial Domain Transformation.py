#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("new.jpg")

#form the filters
#kernel_identity = np.array([[0,0,0],[0,1,0],[0,0,0]])
kernel_3 = np.ones((3,3),dtype=np.float32)/(3*3)
#kernel_11 = np.ones((11,11),dtype=np.float32)/(11*11)
#LPF box
output = cv2.filter2D(img,-1,kernel_3)


#HPF
output2 = img - output


# Gaussian LPF

output3 = cv2.GaussianBlur(img,(5,5),7)


#Median PF (noise reduction)

output4 = cv2.medianBlur(img,5)

plt.figure(figsize=(20,20 ))
plt.subplot(6, 2, 1)
plt.title("Original image",fontsize =20)
plt.axis(False)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(6, 2, 2)
plt.title('Histogram for original picture',fontsize =20)

color = ('b', 'g', 'r')
for channel, col in enumerate(color):
    histr = cv2.calcHist([img], [channel], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])

plt.subplot(6, 2, 3)

plt.title("BOX filter LPF",fontsize =20)
plt.axis(False)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))



plt.subplot(6, 2, 4)
for channel, col in enumerate(color):
    histr = cv2.calcHist([output], [channel], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.title('Histogram for Box Filter Blurred picture',fontsize =20)


plt.subplot(6, 2, 5)

plt.title("HIGH PASS Filter",fontsize =20)
plt.axis(False)
plt.imshow(cv2.cvtColor(output2, cv2.COLOR_BGR2RGB))


plt.subplot(6, 2, 6)
for channel, col in enumerate(color):
    histr = cv2.calcHist([output2], [channel], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.title('Histogram for High Pass Filter picture',fontsize =20)

plt.subplot(6, 2, 7)

plt.title("Gaussian Filter",fontsize =20)
plt.axis(False)
plt.imshow(cv2.cvtColor(output3, cv2.COLOR_BGR2RGB))


plt.subplot(6, 2, 8)
for channel, col in enumerate(color):
    histr = cv2.calcHist([output3], [channel], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.title('Histogram for gaussian lpf picture',fontsize =20)


plt.subplot(6, 2, 9)

plt.title("Median Filter",fontsize =20)
plt.axis(False)
plt.imshow(cv2.cvtColor(output4, cv2.COLOR_BGR2RGB))


plt.subplot(6, 2, 10)
for channel, col in enumerate(color):
    histr = cv2.calcHist([output4], [channel], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.title('Histogram for Median picture',fontsize =20)


plt.show()



# In[ ]:




