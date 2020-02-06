
# coding: utf-8

# In[1]:


#coding = utf-8
import pandas as pd
import numpy as np


# In[2]:


file = pd.read_csv('/Users/suri/Desktop/alldanmu.csv')


# In[3]:


file


# In[4]:


danmu = file['content']


# In[5]:


danmu


# In[6]:


type(danmu)


# In[7]:


danmu = danmu.to_string()


# In[8]:


type(danmu)


# In[9]:


print(danmu)


# In[12]:


import jieba
mydanmu = " ".join(jieba.cut(danmu))


# In[13]:


print(mydanmu)


# In[19]:


from wordcloud import WordCloud
wordcloud = WordCloud(
        font_path="/Users/suri/Desktop/font.ttf",
        width=2048,
        height=1024,
        background_color='white',# 设置背景颜色
        #mask=backgroud_Image,# 设置背景图片
        max_words=1000, # 设置最大现实的字数
        #stopwords=stopwords,# 设置停用词
        max_font_size=400,# 设置字体最大值
        random_state=50,# 设置有多少种随机生成状态，即有多少种配色方案
).generate(mydanmu)

import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


# In[15]:


timelop = file[['content','upcount','timepoint']]


# In[16]:


timelop


# In[17]:


timelop['contentcount'] = 1


# In[29]:


timeana1 = timelop


# In[30]:


timeana = timelop.groupby('timepoint').sum()


# In[31]:


timeana


# In[32]:


timeana = timeana.reset_index()


# In[66]:


print(timeana)


# In[34]:


import matplotlib.pyplot as plt


# In[35]:



l1=plt.plot(timeana['timepoint'],timeana['upcount'],'g--',label='upcount')
plt.title('1st')
plt.xlabel('time',fontproperties='SimHei')
plt.ylabel('count',fontproperties='SimHei')
plt.legend()
plt.show()


# In[36]:


l2=plt.plot(timeana['timepoint'],timeana['contentcount'],'r--',label='danmu')
plt.title('1st')
plt.xlabel('time',fontproperties='SimHei')
plt.ylabel('count',fontproperties='SimHei')
plt.legend()
plt.show()


# In[38]:


timeana1


# In[65]:


bins = np.arange(0, 3000, 20) # fixed bin size
data = timeana1['timepoint']
plt.xlim([min(data)-5, max(data)+5])
 
plt.hist(data, bins=bins, alpha=0.5)
plt.title('1st-comments')
plt.xlabel('time(bin size = 20)')
plt.ylabel('count')
 
plt.show()


# In[46]:


import codecs
import re
import numpy as np
from snownlp import SnowNLP
import matplotlib.pyplot as plt
from snownlp import sentiment
from snownlp.sentiment import Sentiment


# In[47]:


commentana = file['content']


# In[ ]:


import pymysql


# In[48]:


type(commentana)


# In[50]:


commentana.tolist()


# In[62]:


sentimentslist = []
for li in commentana:
    s = SnowNLP(li)
    print(li)
    print(s.sentiments)
    sentimentslist.append(s.sentiments)


# In[60]:


#snowanalysis(commentana)


# In[64]:


plt.hist(sentimentslist,bins=np.arange(0,1,0.01))
plt.show()

