#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


df=pd.DataFrame({'x':[12,20,28,18,29,33,24,45,52,51,53,55,54,65,61,67,69,72,76],
                'y':[39,36,30,52,54,46,55,59,63,70,66,63,58,23,14,8,19,7,78]})


# In[13]:


df.head()


# In[11]:


centroids={i+1:[np.random.randint(0,80),
                np.random.randint(0,80)]
           for i in range(3)}


# In[12]:


plt.scatter(df.x,df.y)
colmap = {1 : 'r', 2: 'g', 3: 'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color = colmap[i])
plt.show()


# In[21]:


#Assignment stage
def assignment(df, centroids):
    for i in centroids.keys():
        #sqrt((x1-x2)^2+(y1-y2)^2)
        #Euclidean distance
        df['distance_from_{}'.format(i)] = np.sqrt((df['x']-centroids[i][0]) ** 2 + 
                                                    (df['y']-centroids[i][1]) ** 2)
    df['closest'] = df.loc[:,'distance_from_1':'distance_from_3'].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x : int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

df = assignment(df, centroids)
df.head()


# In[23]:


plt.scatter(df.x,df.y,color=df['color'],edgecolor='k')
colmap = {1 : 'r', 2: 'g', 3: 'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color = colmap[i])
plt.show()


# In[24]:


#update Stage
import copy
old_centroids=copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x']) 
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k

centroids = update(centroids)


# In[25]:


print(old_centroids)
print(centroids)


# In[27]:


#Repeat Assignment Stage
df = assignment(df,centroids)
centroids = update(centroids)
df = assignment(df,centroids)

plt.scatter(df.x,df.y,color=df['color'],edgecolor='k')
colmap = {1 : 'r', 2: 'g', 3: 'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color = colmap[i])
plt.show()


# In[28]:


df=pd.DataFrame({
    'x':[12,20,28,18,29,33,24,45,52,51,52,55,53,55,61,64,69,72],
    'y':[39,36,30,52,54,46,55,59,63,70,66,63,58,23,14,8,19,7]})


# In[29]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(df)


# In[30]:


centroids = kmeans.cluster_centers_
centroids


# In[31]:


labels = kmeans.predict(df)
labels


# In[32]:


colors = map(lambda x: colmap[x+1], labels)
colors1 = list(colors)

plt.scatter(df['x'],df['y'], color=colors1, edgecolor='k')
for idx,centroid in enumerate(centroids):
    plt.scatter(*centroid ,color=colmap[idx+1])
    
plt.show()


# In[33]:


cost=[]
for i in range(1,11):
    KM = KMeans(n_clusters = i, max_iter = 400)
    KM.fit(df)
    
    cost.append(KM.inertia_)


# In[35]:


plt.plot(np.arange(1,11), np.array(cost))
plt.xlabel('Value of K')
plt.ylabel('Squared Error (Cost)')
plt.show()


# In[ ]:




