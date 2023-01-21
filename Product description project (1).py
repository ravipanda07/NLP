#!/usr/bin/env python
# coding: utf-8

# In[34]:


#!pip install nltk
#!pip install pipelines==0.0.12
#!pip install pipreqs
#!pip install spacy
#!pip install wordcloud
#!pip install sklearn
#!pip install svm
#!pip install wordcloud
#!pip install scikit-learn
#! pip install xgboost
#!pip install contractions
get_ipython().system('pip install textblob')
import nltk.corpus
import numpy as np
import pandas as pd
import pipreqs as pipreqs
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
import re
from spacy.lang.en.tokenizer_exceptions import word
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore') 
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
get_ipython().system('pip install pipelines==0.0.12')


# In[86]:


get_ipython().system('pip install rake-nltk')
from rake_nltk import Rake


# In[12]:


df=pd.read_csv("Downloads/Product_details.csv",error_bad_lines=False)


# In[13]:


df


# In[14]:


df.shape


# In[15]:


df.info()


# In[16]:


df.describe()


# In[17]:


# Visualization some insights from Raw Data
df['Product_Type'].value_counts()
sns.histplot(df['Product_Type'])
plt.show()


# In[18]:


df['Sentiment'].value_counts()
sns.histplot(df['Sentiment'])
plt.show()


# In[19]:


# Plotting Pie Chart
values = df['Product_Type'].value_counts()     # Counting the unique values frequency
labels = df['Product_Type'].unique().tolist()  # Creating the unique value labels
explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0)     # Exploding the first slice


# In[20]:


# Creating the Pie Chart with included exploding slice
plt.pie(values, labels = labels, explode = explode, radius = 1)


# In[21]:


counter = []
for string in df.Product_Description:
    counter.append(string.count(' ') + 1)  # Num of spaces + 1

df['num_words'] = counter  # add the column
df.head(5)


# In[87]:


pd.crosstab(df.Sentiment, df.Product_Type)


# In[89]:


pd.crosstab(df.Sentiment, df.Product_Type).apply(lambda r:r/r.sum(),axis=1)


# In[90]:


import pandas as pd
import matplotlib.pyplot as plt

# creating dataframe
dataFrame = pd.DataFrame({
   "Product_type": ['9', '6', '2', '7', '3', '5','8','1','0','4'],"Counts": [4070, 665, 465,327,300,213,194,59,52,18]
})

# plot a Pie Chart for Registration Price column with label Car column
plt.figure(figsize=(10,8));
plt.pie(dataFrame["Counts"], labels = dataFrame["Product_type"])
plt.title("Product type visualization")
plt.show()


# In[91]:


import pandas as pd
import matplotlib.pyplot as plt

# creating dataframe
dataFrame = pd.DataFrame({
   "Sentiments": ['2','3','1','0'],"Counts": [3765,2089,399,111]
})

# plot a Pie Chart for Registration Price column with label Car column
plt.figure(figsize=(10,8));
plt.pie(dataFrame["Counts"], labels = dataFrame["Sentiments"])
plt.title("Sentiments Analysis")
plt.show()


# In[ ]:





# In[94]:


ratings =df["Sentiment"].value_counts()
numbers = ratings.index
quantity = ratings.values

custom_colors = ["skyblue", "yellowgreen", 'tomato', "blue", "red"]
plt.figure(figsize=(10, 8))
plt.pie(quantity, labels=numbers, colors=custom_colors)
central_circle = plt.Circle((0, 0), 0.5, color='white')
fig = plt.gcf()
fig.gca().add_artist(central_circle)
plt.rc('font', size=12)
plt.title("Distribution of Amazon Product Ratings", fontsize=20)
plt.show()


# In[22]:


# Cleaning the Dataset
# Text Preprocessing Techniques
import re

def cleantext(text):
    text = re.sub(r"â€™", "", text)             # Remove Mentions
    text = re.sub(r"#", "", text)               # Remove Hashtags Symbol
    text = re.sub(r"\w*\d\w*", "", text)        # Remove numbers
    text = re.sub(r"https?:\/\/\S+", "", text)  # Remove The Hyper Link
    text = re.sub(r"______________", "", text)  # Remove _____
    text=re.sub(r"^a-zA-z0-9","",text)
    text=re.sub(r"[^\w\s]","",text)

    return text


# In[25]:


from nltk.tokenize import word_tokenize

df['clean_text'] = df.apply(lambda x: cleantext(x['Product_Description']), axis = 1)
df['clean_text']

# Contractions
import contractions
df['no_contract'] = df['clean_text'].apply(lambda x: [contractions.fix(word) for word in x.split()])
df['no_contract']

# Tokenization
from nltk.tokenize import word_tokenize
df['tokenized']  = df['clean_text'].apply(word_tokenize)

# Lower Case Conversion
df['lower'] = df['tokenized'].apply(lambda x: [word.lower() for word in x])

# Joining df['lower']
df['lower'] = [' '.join(map(str,i)) for i in df['lower']]

df.head(5)

# Stopwords
import nltk
from nltk.corpus import stopwords
from wordcloud import STOPWORDS

stopwords = nltk.corpus.stopwords.words('english')
newstopwords = ["SXSW", "sxsw", "link","iPhone", "upad", "Apple popup" , "RT mention", "RT", "rt", "sxsw sxsw", "Google", "DesignerÛªs" , "link sxsw", "iPad launch", "Social Network", "sxsw apple", "amp","mention google", "via mention", "called circles" , "popup store", "link via", "sxsw sxswi", "downtown austin", "ûïmention" , "sxswi", "marissa mayer", "an iPad", "Circles Possibly", "Austin for","new iPad", "iPad at", "temporary store" , "New UberSocial", "Apple i", "Apple", "popup store", "in Austin", "Called Circles", "Network Called", "Social Network", "Austin","iPad", "Apple Store", "New Social", "sxswÛ", "Facebook", "Circles Possibly", "downtown Austin", "ipad design", "designerûªs", "Marissa Mayer"] + list(stopwords)
list(newstopwords)

stops = r'\b({})\b'.format('|'.join(newstopwords))


# In[26]:


df['nostop'] = df['lower'].str.replace(stops, '').str.replace('\s+', ' ')
df.head()


# In[29]:


# Now Generating Wordcloud using df['nostop']

#wordc = ' '.join(df['nostop'])
#wordcloud = WordCloud(width = 3000, height = 1500, background_color = 'black', stopwords = newstopwords, max_words = 400, colormap='Set2').generate(wordc)
#plot_cloud(wordcloud)


# In[30]:


# Stemming
from nltk.stem import PorterStemmer
st = PorterStemmer()


# In[31]:


# stemming ever word
df['stemmed'] = df['nostop'].apply(lambda x: ' '.join([st.stem(word) for word in x.split()]))


# In[32]:


df['stemmed']


# In[35]:


# Lemmatization
from textblob import Word
df['lemma'] = df['stemmed'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# In[36]:


stops = r'\b({})\b'.format('|'.join(newstopwords))
df['lemma'] = df['lemma'].str.replace(stops, '').str.replace('\s+', ' ')
df['lemma'].head(5)


# In[37]:


# Polarity and Subjectivity
from textblob import TextBlob
df['Polarity'] = df['Product_Description'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['Subjectivity'] = df['Product_Description'].apply(lambda x: TextBlob(x).sentiment.subjectivity)


# In[38]:


# Function to analyze the reviews
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


# In[39]:


df['Analysis'] = df['Polarity'].apply(getAnalysis)


# In[40]:


# plotting graph for Polarity (Negative, Neutral, Positive)
Negative_senti = df[df['Polarity']>0]
Neutral_senti = df[df['Polarity']==0]
Positive_senti = df[df['Polarity']<0]


# In[41]:


df['Analysis'] = df['Polarity'].apply(getAnalysis)


# In[42]:


df['Analysis'].value_counts().plot(kind='bar')   # Bar Plot
sns.lmplot (x='Polarity', y='Sentiment', data=df, fit_reg=True)   # Scatter Plot


# In[43]:


# Polarity Distribution
plt.figure(figsize=(20,10))
plt.margins(0.04)
plt.xlabel('Polarity', fontsize=15)
plt.xticks(fontsize=20)
plt.ylabel('Frequency', fontsize=15)
plt.yticks(fontsize=20)
plt.hist(df['Polarity'], bins=40)
plt.title('Polarity Distribution', fontsize=20)
plt.show()


# In[44]:


# Positive reviews Wordcloud
from wordcloud import WordCloud
wc = WordCloud(width=3000, height=1500, min_font_size=10, max_words=300, stopwords=newstopwords, background_color='black')
Positive = wc.generate(df[df['Polarity']>0]['Product_Description'].str.cat(sep=""))

plt.figure(figsize=(10,10))
plt.imshow(Positive)
plt.title('Positive Reviews')
plt.show()


# In[45]:


# Negative Reviews Wordcloud
Negative=wc.generate(df[df['Polarity']<0]['Product_Description'].str.cat(sep=""))

plt.figure(figsize=(10,10))
plt.imshow(Negative)
plt.title('Negative Reviews')
plt.show()


# In[46]:


# Neutral Reviews Wordcloud
Neutral = wc.generate(df[df['Polarity']==0]['Product_Description'].str.cat(sep=""))

plt.figure(figsize=(10,10))
plt.imshow(Neutral)
plt.title('Neutral Reviews')
plt.show()


# In[47]:


# Pivot Table
df.pivot_table(columns=['Product_Type'], values=['Polarity','Subjectivity'])

df.pivot_table(columns=['Sentiment'], values=['Polarity','Subjectivity'])

temp = df[['num_words','Product_Type','Sentiment','Polarity','Subjectivity']]
temp


# In[48]:


# feature Extraction
Negative_senti.head(5)
Neutral_senti.head(5)
Positive_senti.head(5)


# In[49]:


# PCA Principal Component Analysis
#scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(temp)
#tranforming the values
scaled_data = scaler.transform(temp)
#implementing PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(scaled_data)


# In[50]:


x_pca = pca.transform(scaled_data)
#identifying the number of columns
print("The shape of original dataset:",scaled_data.shape)
print("The shape of dataset after implementing PCA:",x_pca.shape)
x_pca


# In[51]:


# words into vector -BOW,Tf-idf,Wordevec
# BOW/Count Vectorization on positive sentiments/reviews

Positive_senti = [lemma.strip() for lemma in Positive_senti.lemma] # remove both the leading and the trailing characters
Positive_senti = [lemma for lemma in Positive_senti if lemma] # removes empty strings, because they are considered in Python as False
Positive_senti[0:10]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(Positive_senti)

word_freq_df = pd.DataFrame({'term': vectorizer.get_feature_names_out(), 'occurrences':np.asarray(X.sum(axis=0)).ravel().tolist()})
word_freq_df.sort_values('occurrences',ascending=False)


# In[52]:


# BOW on Negative words
Negative_senti=Negative_senti['lemma']
Negative_senti=pd.DataFrame(data=Negative_senti)

Negative_senti = [lemma.strip() for lemma in Negative_senti.lemma] # remove both the leading and the trailing characters
Negative_senti = [lemma for lemma in Negative_senti if lemma] # removes empty strings, because they are considered in Python as False
Negative_senti[0:10]


# In[53]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(Negative_senti)

print(vectorizer.vocabulary_)


# In[54]:


word_freq_df = pd.DataFrame({'term': vectorizer.get_feature_names_out(), 'occurrences':np.asarray(X.sum(axis=0)).ravel().tolist()})
word_freq_df.sort_values('occurrences',ascending=False)


# In[55]:


# TFidf Vectorizer on Positive Reviews
from sklearn.feature_extraction.text import TfidfVectorizer


# In[56]:


vectorizer_n_gram_max_features = TfidfVectorizer(norm="l2",analyzer='word', ngram_range=(1,3), max_features = 5000)
tf_idf_matrix_n_gram_max_features =vectorizer_n_gram_max_features.fit_transform(Positive_senti)


# In[57]:


print(vectorizer_n_gram_max_features.get_feature_names_out())
print(tf_idf_matrix_n_gram_max_features.toarray())


# In[58]:


# TFidf Vectorizer on Negative Reviews
vectorizer_n_gram_max_features = TfidfVectorizer(norm="l2",analyzer='word', ngram_range=(1,3), max_features = 5000)
tf_idf_matrix_n_gram_max_features =vectorizer_n_gram_max_features.fit_transform(Negative_senti)


# In[59]:


print(vectorizer_n_gram_max_features.get_feature_names_out())
print(tf_idf_matrix_n_gram_max_features.toarray())

df.head()
temp.head()


# In[60]:


corpus = df['lemma'].tolist()
corpus


# In[61]:


df['Analysis'] = df['Analysis'].replace({'Negative': -1})
df['Analysis'] = df['Analysis'].replace({'Positive': 1})
df['Analysis'] = df['Analysis'].replace({'Neutral': 0})

senti_into_number_form=df['Analysis']

senti_into_number_form.head(50)


# In[62]:


# Corpus converted into array using TFidf
vectorizer4 = TfidfVectorizer(max_features=8000)
idf = vectorizer4.fit_transform(corpus).toarray()
idf


# In[63]:


import pickle
pickle_out=open('vectorizer4.pkl','wb')
pickle.dump(vectorizer4,pickle_out)
pickle_out.close()


# In[64]:


xtfidf = pd.DataFrame(idf)
xtfidf


# In[65]:


#Target var
ytfidf=df['Sentiment']
ytfidf.value_counts() # checking whether data is balanced or imbalanced


# In[66]:


# Test and Split Training Data
from sklearn.model_selection import train_test_split

x_traintfidf, x_testtfidf,y_traintfidf,y_testtfidf = train_test_split(xtfidf,ytfidf, test_size=0.33,random_state=0)
x_traintfidf.shape,y_traintfidf.shape, x_testtfidf.shape,y_testtfidf.shape


# In[67]:


# Balancing the splitted (tfidf) data using SMOTE method
#from imblearn.over_sampling import SMOTE

#upsample = SMOTE()
#x_traintfidf1, y_traintfidf1 = upsample.fit_resample(x_traintfidf, y_traintfidf)


# In[68]:


get_ipython().system('pip install klib')


# In[69]:


get_ipython().system('pip install k-fold-imblearn')


# In[71]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
model_Lg=LogisticRegression()
#Support Vector Classification
from sklearn.svm import SVC
svm_model=SVC()
#NB
from sklearn.naive_bayes import MultinomialNB
NB_model=MultinomialNB()
#Random ForestClassifier
from sklearn.ensemble import RandomForestClassifier
RF_model=RandomForestClassifier()
#XGBoost Classifier
from xgboost import XGBClassifier
XGBoost_model=XGBClassifier()
#AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
AB_model=AdaBoostClassifier()


# In[72]:


lstmodel=[model_Lg,svm_model,NB_model,RF_model,XGBoost_model,AB_model,]
from sklearn.metrics import ConfusionMatrixDisplay,classification_report
for i in lstmodel:
 print(i)
 i.fit(x_traintfidf,y_traintfidf)
 y_pred=i.predict(x_testtfidf)
 print('********************************************************************************')
 print(classification_report(y_testtfidf,y_pred))
 print('*****************************************************************************')


# In[75]:


get_ipython().system('pip install imbalanced-learn')


# In[155]:


# Balancing the splitted (tfidf) data using SMOTE method
#from imblearn.over_sampling import SMOTE
#upsample=SMOTE()
#x_traintfidf1, y_traintfidf1 = upsample.fit_resample(x_traintfidf,y_traintfidf)


# In[73]:


#target y
ytfidf1=df['Sentiment']
ytfidf1.value_counts()


# In[74]:


# Model building with balanced data using Tfidf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[75]:


#1. Logistic Regression
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(x_traintfidf,y_traintfidf)
y_pred_test_lr=lr.predict(x_testtfidf)
y_pred_train_lr=lr.predict(x_traintfidf)


# In[76]:


# accuracy score
accuracy_train_LR=accuracy_score(y_traintfidf,y_pred_train_lr)*100
accuracy_test_LR= accuracy_score(y_testtfidf, y_pred_test_lr) * 100
print('Accuracy of Training data =',accuracy_train_LR)
print("Accuracy of Test data =", accuracy_test_LR)
print(classification_report(y_testtfidf, y_pred_test_lr))


# In[77]:


#2. Random Forest
from sklearn.ensemble import RandomForestClassifier
RF= RandomForestClassifier()
RF.fit(x_traintfidf,y_traintfidf)
y_pred_test_RF=RF.predict(x_testtfidf)
y_pred_train_RF=RF.predict(x_traintfidf)

# accuracy score
accuracy_train_RF=accuracy_score(y_traintfidf,y_pred_train_RF)*100
accuracy_test_RF= accuracy_score(y_testtfidf, y_pred_test_RF) * 100
print('Accuracy of Training data =',accuracy_train_RF)
print("Accuracy of Test data =", accuracy_test_RF)

print(classification_report(y_testtfidf, y_pred_test_RF))


# In[78]:


#SVM
from sklearn.svm import LinearSVC
SVM= LinearSVC()
SVM.fit(x_traintfidf,y_traintfidf)
y_pred_test_SVM=SVM.predict(x_testtfidf)
y_pred_train_SVM=SVM.predict(x_traintfidf)

print(classification_report(y_testtfidf, y_pred_test_SVM))

accuracy_train_SVM=accuracy_score(y_traintfidf,y_pred_train_SVM)*100
accuracy_test_SVM= accuracy_score(y_testtfidf, y_pred_test_SVM) * 100
print('Accuracy of Training data =',accuracy_train_SVM)
print("Accuracy of Test data =", accuracy_test_SVM)


# In[79]:


#4.Naive Bayes classifier for multinomial models

from sklearn.naive_bayes import MultinomialNB
MULT_NB= MultinomialNB()
MULT_NB.fit(x_traintfidf,y_traintfidf)
y_pred_test_mult_nb=MULT_NB.predict(x_testtfidf)
y_pred_train_mult_nb=MULT_NB.predict(x_traintfidf)

accuracy_train_MULT_NB=accuracy_score(y_traintfidf,y_pred_train_mult_nb)*100
accuracy_test_MULT_NB= accuracy_score(y_testtfidf, y_pred_test_mult_nb) * 100
print('Accuracy of Training data =',accuracy_train_MULT_NB)
print("Accuracy of Test data =", accuracy_test_MULT_NB)
print(classification_report(y_testtfidf, y_pred_test_mult_nb))


# In[80]:


# 5AdaBoost classifier on tfidf features
from sklearn.ensemble import AdaBoostClassifier
ADA= AdaBoostClassifier()
ADA.fit(x_traintfidf,y_traintfidf)
y_pred_test_ada=ADA.predict(x_testtfidf)
y_pred_train_ada=ADA.predict(x_traintfidf)

accuracy_train_ADA=accuracy_score(y_traintfidf,y_pred_train_ada)*100
accuracy_test_ADA= accuracy_score(y_testtfidf, y_pred_test_ada) * 100
print('Accuracy of Training data =',accuracy_train_ADA)
print("Accuracy of Test data =", accuracy_test_ADA)
print(classification_report(y_testtfidf, y_pred_test_ada))


# In[81]:


# 6.KNN
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

KNN= AdaBoostClassifier()
KNN.fit(x_traintfidf,y_traintfidf)
y_pred_test_KNN=KNN.predict(x_testtfidf)
y_pred_train_KNN=KNN.predict(x_traintfidf)

accuracy_train_KNN=accuracy_score(y_traintfidf,y_pred_train_ada)*100
accuracy_test_KNN= accuracy_score(y_testtfidf, y_pred_test_ada) * 100
print('Accuracy of Training data =',accuracy_train_KNN)
print("Accuracy of Test data =", accuracy_test_KNN)
print(classification_report(y_testtfidf, y_pred_test_KNN))


# In[82]:


#7.XG-Boost
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

XGB = XGBClassifier()
XGB.fit(x_traintfidf,y_traintfidf)

# make predictions for test data
y_pred_test_XGB = XGB.predict(x_testtfidf)
y_pred_train_XGB = XGB.predict(x_traintfidf)

accuracy_train_XGB=accuracy_score(y_traintfidf,y_pred_train_ada)*100
accuracy_test_XGB= accuracy_score(y_testtfidf, y_pred_test_ada) * 100
print('Accuracy of Training data =',accuracy_train_XGB)
print("Accuracy of Test data =", accuracy_test_XGB)
print(classification_report(y_testtfidf, y_pred_test_XGB))


# In[84]:


#Saving the Best Model
filename = 'saved_lr_model.pkl'
pickle.dump(lr,open(filename,'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.predict(x_testtfidf)


# In[ ]:





# In[ ]:




