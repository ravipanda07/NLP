import nltk
nltk.download('wordnet')
import warnings
from nltk.corpus import stopwords, wordnet
import streamlit as st
nltk.download('stopwords')
import re
from rake_nltk import Rake
import pickle

import numpy as np
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Warnings ignore
warnings.filterwarnings(action='ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)


# loading the trained model
pickle_in = open('saved_lr_model.pkl', 'rb')
model = pickle.load(pickle_in)


pickle_in = open('vectorizer4.pkl', 'rb')
vectorizer = pickle.load(pickle_in)
# Title of the application
st.title('Analysis of the Amazon data\n', )

input_text = st.text_area("Enter review", height=50)

# Sidebar options
option = st.sidebar.selectbox('Selecting one', ['Sentiment Analysis', 'Check Keywords', 'Word Cloud'])
st.set_option('deprecation.showfileUploaderEncoding', False)


if option == "Sentiment Analysis":

    if st.button("Predict sentiment"):
        wordnet = WordNetLemmatizer()
        text = re.sub('[^A-za-z0-9]', ' ', input_text)
        text = text.lower()
        text = text.split(' ')
        text = [wordnet.lemmatize(word) for word in text if word not in (stopwords.words('english'))]
        text = ' '.join(text)
        pickle_in = open('saved_lr_model.pkl', 'rb')
        model = pickle.load(pickle_in)
        pickle_in = open('vectorizer4.pkl', 'rb')
        vectorizer = pickle.load(pickle_in)
        transformed_input = vectorizer.transform([text])

        if model.predict(transformed_input) == -1:
            st.write("Input review has  Sentiment = 0")
        elif model.predict(transformed_input) == 1:
            st.write("Input review has  Sentiment = 1")
        elif model.predict(transformed_input) == 1:
            st.write("Input review has  Sentiment = 2")
        else:
            st.write(" Input review has  Sentiment = 3")
elif option == "Check Keywords":
    st.header("Check Keywords")
    if st.button("Check Keywords"):
        
        r = Rake(language='english')
        r.extract_keywords_from_text(input_text)
        # Get the important phrases
        phrases = r.get_ranked_phrases()
        # Get the important phrases
        phrases = r.get_ranked_phrases()
        # Display the important phrases
        st.write("Below are the keywords:")
        for i, p in enumerate(phrases):
            st.write(i + 1, p)
elif option == "Word Cloud":
    st.header("Word cloud")
    if st.button("Wordcloud"):
        wordnet = WordNetLemmatizer()
        text = re.sub('[^A-za-z0-9]', ' ', input_text)
        text = text.lower()
        text = text.split(' ')
        text = [wordnet.lemmatize(word) for word in text if word not in (stopwords.words('english'))]
        text = ' '.join(text)
        wordcloud = WordCloud().generate(text)
        plt.figure(figsize=(40, 30))
        plt.imshow(wordcloud)
        plt.axis("off")
          
        st.pyplot()
