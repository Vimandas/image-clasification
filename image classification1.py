
# coding: utf-8

# In[2]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
 
# load the MNIST digits dataset
mnist = datasets.load_digits()
(X_train,X_test,y_train, y_test) = train_test_split(np.array(mnist.data),
    mnist.target, test_size=0.25, random_state=42)


# In[3]:


model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)


# In[5]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)


# In[6]:



import matplotlib
import matplotlib.pyplot as plt
i=10
some_digit = X_train[i]

some_digit_image = some_digit.reshape(8, 8)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
           interpolation="nearest")
plt.axis("off")
print(y_train[i])


# In[7]:


mnist['DESCR']

