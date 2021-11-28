#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic('matplotlib', 'inline')

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# In[2]:



def plot_images(img1, img2, title1="", title2=""):
    fig = plt.figure(figsize=[15,15])
    ax1 = fig.add_subplot(121)
    ax1.imshow(img1, cmap="gray")
    ax1.set(xticks=[], yticks=[], title=title1)

    ax2 = fig.add_subplot(122)
    ax2.imshow(img2, cmap="gray")
    ax2.set(xticks=[], yticks=[], title=title2)


# In[3]:


path="./t2.png"
image=cv2.imread(path)
plot_images(image, image)


# In[4]:


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plot_images(image, gray)


# In[5]:


blur=cv2.bilateralFilter(gray, 11, 50, 50)
plot_images(gray, blur)


# In[6]:


edges=cv2.Canny(blur,30,200)
plot_images(blur, edges)


# In[7]:


cntr,new = cv2.findContours(edges.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


# In[8]:


image_copy=image.copy()


# In[9]:


_=cv2.drawContours(image_copy,cntr,-1,(225,0,225),2)


# In[10]:


plot_images(edges, image_copy)


# In[11]:


print(len(cntr)) # length of Contours or how many Contours are there in Image


# In[12]:


cntr = sorted(cntr, key=cv2.contourArea, reverse=True)[:10]


# In[13]:


image_copy = image.copy()
_ = cv2.drawContours(image_copy, cntr, -1, (255,0,255),2)


# In[14]:


plot_images(image, image_copy)


# In[15]:


plate = None
for c in cntr:
    perimeter = cv2.arcLength(c, True)
    edges_count = cv2.approxPolyDP(c, 10, True)
    if len(edges_count) == 4:
        x,y,w,h = cv2.boundingRect(c)
        plate = image[y:y+h, x:x+w]
        break

cv2.imwrite("plate.png", plate)


# In[16]:


gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
plot_images(plate, gray_plate)


# In[17]:


text = pytesseract.image_to_string(gray_plate, lang="eng")
print(text)


# In[ ]:




