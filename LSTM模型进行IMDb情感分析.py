#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
import urllib.request
import os
import tarfile
from tensorflow.keras.datasets import imdb


# In[3]:


(x_train,y_train),(x_test,y_test) = keras.datasets.imdb.load_data()


# In[4]:


import re
def rm_tags(text):
    re_tag=re.compile(r'<[^>]+>')
    return re_tag.sub('',text)


# In[5]:


import os
def read_files(filetype):
    path="data/aclImdb/"
    file_list=[]
    
    positive_path=path+filetype+"/pos/"
    for f in os.listdir(positive_path):
        file_list+=[positive_path+f]
        
    negative_path=path+filetype+"/neg/"
    for f in os.listdir(negative_path):
        file_list+=[negative_path+f]
        
    print('read',filetype,'files:',len(file_list))
    
    all_labels=([1]*12500+[0]*12500)
    
    all_texts=[]
    for fi in file_list:
        with open(fi,encoding='utf8') as file_input:
            all_texts+=[rm_tags("".join(file_input.readlines()))]
            
    return all_labels,all_texts


# In[6]:


y_train,train_text=read_files("train")


# In[7]:


y_test,test_text=read_files("test")


# In[8]:


token = Tokenizer(num_words=2000)
token.fit_on_texts(train_text)


# In[9]:


x_train_seq=token.texts_to_sequences(train_text)
x_test_seq=token.texts_to_sequences(test_text)


# In[10]:


x_train=sequence.pad_sequences(x_train_seq,maxlen=100)
x_test =sequence.pad_sequences(x_test_seq,maxlen=100)


# In[11]:


n_timesteps = 10 


# In[12]:


model = Sequential()


# In[13]:


model.add(Embedding(output_dim=32,
                    input_dim=2000,
                   input_length=100))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32,return_sequences=True),input_shape=(n_timesteps, 1)))     
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))


# In[14]:


model.summary() 


# In[15]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[16]:


y_train=np.array(y_train)  
y_test=np.array(y_test) 


# In[17]:


train_history=model.fit(x_train,y_train,batch_size=100,
                  epochs=10,verbose=2,validation_split=0.2)


# In[18]:


scores = model.evaluate(x_test,y_test,verbose=1)
scores[1]


# In[ ]:




