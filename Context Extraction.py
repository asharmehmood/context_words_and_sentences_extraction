#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('ls')


# In[42]:


import PyPDF2

file = open('ogdc_2018.pdf', 'rb')
# creating a pdf reader object
fileReader = PyPDF2.PdfFileReader(file)
page = fileReader.getPage(14)e.extractText().replace('\n',' ')


# In[42]:


from pdf2image import convert_from_path
# 53-64

page =53
pages = convert_from_path('ogdc_2018.pdf', first_page = page, last_page = page, grayscale=False)

pages[0].save('out.jpg', 'JPEG')
    
get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('out.jpg')
imgplot = plt.imshow(img)
plt.show()


# In[43]:


import cv2 
import pytesseract
import re

img = cv2.imread('out.jpg')

custom_config = r'--oem 3 --psm 1'
text = pytesseract.image_to_string(img, config=custom_config)
txt=text.replace('\n',' ')
# txt=txt[:825]
print(txt)
# txt.split('.')
# re.split('.',txt)


# In[129]:



import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math

nltk.download("popular")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter



def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def convert_lower_case(data):
    return np.char.lower(data)

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def remNumAndSmallWords(data):
    data=data.replace(" ", "")
    if data.replace(".", "", 1).isdigit() or len(data)<3:
        return ""
    else:
        return data
    
def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = remove_punctuation(data)
    data = remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one
    data = remNumAndSmallWords(data)
    return data


# In[79]:


get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pdf2image import convert_from_path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import cv2 
import pytesseract
import re

textList=[]
page = [34,45,46,51,53,54,55,56,57,58,59,60,61,62,63,64,79,80]
for pg in page:
    print(pg)
    pages = convert_from_path('ogdc_2018.pdf', first_page = pg, last_page = pg, grayscale=False)
    pages[0].save('out.jpg', 'JPEG')
    
    img=mpimg.imread('out.jpg')
    imgplot = plt.imshow(img)
    plt.show()

    img = cv2.imread('out.jpg')
    custom_config = r'--oem 3 --psm 1'
    text = pytesseract.image_to_string(img, config=custom_config)
    txt=text.replace('\n',' ')
    if pg==45:
        finalTxt=txt[:825]
        textList.append(finalTxt)
    elif pg==53 or pg==79:
        finalTxt=txt
    elif pg>53 and pg<64:
        finalTxt+=txt
    elif pg==64 or pg==80:
        finalTxt+=txt
        textList.append(finalTxt)
    else:
        finalTxt=txt
        textList.append(finalTxt)
print(textList)


# In[138]:


newTextList=[]
newPara=''
for para in textList:
    wordList=para.split(' ')
    newPara=''
    for word in wordList:
        newPara+=" "+str(preprocess(word))
#         print(preprocess(word))
    newTextList.append(newPara)


# In[139]:


print(newTextList)


# In[140]:


# print(newTextList)
vectorizer = TfidfVectorizer(ngram_range=(1,3))
vectors = vectorizer.fit_transform(newTextList)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
df


# In[141]:


import numpy as np

nlargest = 20
topnlocs = np.argsort(-df.values, axis=1)[:, 0:nlargest]
topnlocs


# In[145]:


i=0
cols=df.columns.values.tolist()
sections=['RISK AND OPPORTUNITY REPORT','DUPONT ANALYSIS','SHARE PRICE SENSITIVITY ANALYSIS','MANAGING DIRECTOR’S REVIEW','DIRECTORS’ REPORT','PRINCIPAL RISKS/UNCERTAINTIES AND MITIGATION MEASURES']
context_list=[]
para_list=[]
for arr in topnlocs:
    para_list=[]
    for val in arr:
       para_list.append(cols[val])
    context_list.append(para_list)
    i+=1
print(context_list)


# In[182]:


par=textList[0].replace(';','.').split('.')
cwords=context_list[0]
sentList=[]
sentScore=[]
cnt=0
for pr in par:
    cnt=0
    for wrd in cwords:
        if wrd in pr:
            cnt+=1
    sentScore.append(cnt)
# print(par)   
# sortedSent=sorted(sentScore, reverse=True)
# print(sortedSent)
print(len(sentScore),sentScore)
loc=[]
ilocc=0
for itr in sentScore:
    if itr>=5:
        loc.append(ilocc)
    ilocc+=1
print(loc)
for sents in loc:
    print(par[sents])

