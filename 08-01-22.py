#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.sparse import csr_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Bollywood songs.csv')
df


# In[3]:


df = df.iloc[: , 1:]


# In[4]:


df


# In[5]:


df.rename(columns={'Song-Name' : 'songs',
                        'Singer/Artists':'Singer',
                        'Album/Movie':'Album',
                        'User-Rating':'Ratings'}, inplace=True)
df


# In[6]:


df['Genre'] = df['Genre'].apply(lambda x:x.replace("Bollywood"," "))
df['Ratings'] =df['Ratings'].apply(lambda x:x.replace("/10"," "))


# In[7]:


df


# In[8]:


df.shape


# In[9]:


#df.info()


# In[10]:


df.isna().sum()


# In[11]:


df.Ratings.unique()


# In[12]:


df['Ratings'] = pd.to_numeric(df['Ratings'], errors='coerce')


# In[13]:


mean_value = df.Ratings.mean()
mean_value


# In[14]:


df.Ratings.fillna(value = mean_value, inplace=True)
df


# In[15]:


#print('Minimum rating is: ' ,(df.Ratings.min()))
#print('Maximum rating is: ' ,(df.Ratings.max()))


# In[16]:


#df.dtypes


# In[17]:


df[df.duplicated()].shape


# In[18]:


df[df.duplicated()]


# In[19]:


df1 = df.drop_duplicates()
df1


# In[20]:


df1.shape


# In[22]:


# Check the distribution of the rating
plt.figure(figsize=(10,4))
df1['Ratings'].hist(bins=70)


# In[23]:


#count how many rows we have by song, we show only the ten more popular songs 
ten_pop_songs = df1.groupby('Album')['Ratings'].count().reset_index().sort_values(['Ratings', 'Album'], ascending = [0,1])
ten_pop_songs['percentage']  = round(ten_pop_songs['Ratings'].div(ten_pop_songs['Ratings'].sum())*100, 2)


# In[24]:


ten_pop_songs= ten_pop_songs[:10]
ten_pop_songs


# In[25]:


labels = ten_pop_songs['Album'].tolist()
counts = ten_pop_songs['Ratings'].tolist()


# In[26]:


plt.figure()
sns.barplot(x=counts, y=labels, palette='Set3')
sns.despine(left=True, bottom=True)


# In[27]:


#count how many rows we have by artist name, we show only the ten more popular artist singer
ten_pop_artists  = df1.groupby(['Singer'])['Album'].count().reset_index().sort_values(['Album', 'Singer'], 
                                                                                                ascending = [0,1])


# In[28]:


ten_pop_artists = ten_pop_artists[:10]
ten_pop_artists


# In[29]:


plt.figure()
labels = ten_pop_artists['Singer'].tolist()
counts = ten_pop_artists['Album'].tolist()
sns.barplot(x=counts, y=labels, palette='Set2')
sns.despine(left=True, bottom=True)


# In[30]:


df1.head()


# In[31]:


df1.shape


# In[32]:


df1.Genre.value_counts()


# In[33]:


df_songs_features = df1.pivot_table(index='songs', columns='Genre', values='Ratings').fillna(0)
df_songs_features


# In[34]:


X = csr_matrix(df_songs_features.values)


# In[35]:


df1.head()


# In[36]:


num = len(df1['songs'].unique())
song_mapper = dict(zip(np.unique(df1["songs"]), list(range(num))))


# In[37]:


song_inv_mapper = dict(zip(list(range(num)), np.unique(df1["songs"])))


# In[38]:


def find_similar_songs(new_song, X, k, metric='cosine', show_distance=False):
    neighbour_ids = []
    song_ind = song_mapper[new_song]
    song_vec = X[song_ind]
    k+=1    
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    song_vec = song_vec.reshape(1,-1)
    neighbour = kNN.kneighbors(song_vec, return_distance=show_distance)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(song_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids


# In[39]:


find_similar_songs(new_song='Aankh Marey',X=X, k=10)


# In[40]:


find_similar_songs(new_song='Coca Cola',X=X, k=10)


# In[ ]:




