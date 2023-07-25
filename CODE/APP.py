import streamlit as st
from PIL import Image
from datetime import datetime, timedelta

import pandas as pd
#import matplotlib.pyplot as plt
import pickle
import joblib
import os
from io import StringIO

import re
#import string
from operator import itemgetter
import gensim

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


# Load the model :
model_folder = os.path.join(os.path.dirname(__file__), '..', 'MODEL')
model_path = os.path.join(model_folder, 'lda_model_sym_wiki.pkl')
with open(model_path, 'rb') as file:
    lda_model = joblib.load(file) # or pickle.load(file)

# Load the image : 
image_folder = os.path.join(os.path.dirname(__file__), '..', 'DOC')
image_path = os.path.join(image_folder, 'topic_modeling_image.png')
image = Image.open(image_path) 


# 1/ Clean article with regex :
def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def clean_with_regex(article_txt) :
    if not article_txt == None:  
        # Extracting the text portion from the article                                              
        article_txt = article_txt[ : article_txt.find("==")]

        # remove text written between double curly braces
        article_txt = re.sub(r"{{.*}}","",article_txt)

        # remove file attachments
        article_txt = re.sub(r"\[\[File:.*\]\]","",article_txt)

        # remove Image attachments
        article_txt = re.sub(r"\[\[Image:.*\]\]","",article_txt)

        # remove unwanted lines starting from special characters
        article_txt = re.sub(r"\n: \'\'.*","",article_txt)
        article_txt = re.sub(r"\n!.*","",article_txt)
        article_txt = re.sub(r"^:\'\'.*","",article_txt)

        #  remove non-breaking space symbols
        article_txt = re.sub(r"&nbsp","",article_txt)

        # remove URLs link
        article_txt = re.sub(r"http\S+","",article_txt)

        # remove digits from text
        article_txt = re.sub(r"\d+","",article_txt)   

        # remove text written between small braces
        article_txt = re.sub(r"\(.*\)","",article_txt)

        # remove sentence which tells category of article
        article_txt = re.sub(r"Category:.*","",article_txt)

        # remove the sentences inside infobox or taxobox
        article_txt = re.sub(r"\| .*","",article_txt)
        article_txt = re.sub(r"\n\|.*","",article_txt)
        article_txt = re.sub(r"\n \|.*","",article_txt)
        article_txt = re.sub(r".* \|\n","",article_txt)
        article_txt = re.sub(r".*\|\n","",article_txt)

        # remove infobox or taxobox
        article_txt = re.sub(r"{{Infobox.*","",article_txt)
        article_txt = re.sub(r"{{infobox.*","",article_txt)
        article_txt = re.sub(r"{{taxobox.*","",article_txt)
        article_txt = re.sub(r"{{Taxobox.*","",article_txt)
        article_txt = re.sub(r"{{ Infobox.*","",article_txt)
        article_txt = re.sub(r"{{ infobox.*","",article_txt)
        article_txt = re.sub(r"{{ taxobox.*","",article_txt)
        article_txt = re.sub(r"{{ Taxobox.*","",article_txt)

        # remove lines starting from *
        article_txt = re.sub(r"\* .*","",article_txt)

        # remove text written between angle bracket
        article_txt = re.sub(r"<.*>","",article_txt)

        # remove new line character
        article_txt = re.sub(r"\n","",article_txt)  

        # replace all punctuations with space
        article_txt = re.sub(r"\!|\"|\#|\$|\%|\&|\'|\(|\)|\*|\+|\,|\-|\.|\/|\:|\;|\<|\=|\>|\?|\@|\[|\\|\]|\^|\_|\`|\{|\||\}|\~"," ",article_txt)

        # replace consecutive multiple space with single space
        article_txt = re.sub(r" +"," ",article_txt)

        # replace non-breaking space with regular space
        article_txt = article_txt.replace(u'\xa0', u' ')
        
        if is_ascii(article_txt):
            return article_txt
        else :
            return ""
        
              
# 2/ Clean article with stop word and lemmatization :
# Function to remove stop words from sentences & lemmatize words. (pass the article text as string "doc")
stop = set(stopwords.words('english'))
#exclude = set(string.punctuation) #remove punctuation, but useless here.
lemma = WordNetLemmatizer()

def clean_with_stopword_lemmatization(doc):
    
    # remove stop words & punctuation, and lemmatize words
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    #p_free  = ''.join(ch for ch in stop_free if ch not in exclude) #remove punctuation, but useless here because they are already deleted.
    normalized = " ".join(lemma.lemmatize(word,'v') for word in stop_free.split())
    x = normalized.split()
    
    # only take words which are greater than 2 characters
    y = [s for s in x if len(s) > 2]
    return y        
        
                 
# Document clustering :
def cluster_similar_documents(doc):
    #rename topics (unknown if the topic looks neutral):
    topics = ['Education and International Companies',
            'Energy and Natural Elements',
            'Politics and Government in the United States',
            'Language, History, and Culture',
            'Television Series and Fictional Characters',
            'Historical Periods in Europe',
            'Computer Systems and Software',
            'German Royalty and Art',
            'Film Awards and Performances',
            'Military and Aircrafts',
            'Human Rights and Social Movements',
            'Demographics and Geography',
            'Music and Record Releases',
            'Geographic Locations and Parks',
            'Biological Processes and Chemicals',
            'Wars and Battles',
            'British Universities and Schools',
            'Titles and Kingdoms',
            'Sports and Football Leagues',
            'Plant and Species',
            'Colors and Materials',
            'Mathematical Terms and Concepts',
            'Geographic Features and Rivers']


    #Pre-processing :
    doc = clean_with_regex(doc)
    doc = doc if (doc != None) & (len(doc) > 150) else ""
    doc = clean_with_stopword_lemmatization(doc)
    
    #doc_bow
    doc_bow = lda_model.id2word.doc2bow(doc)
    
    #doc_topics
    doc_topics = lda_model.get_document_topics(doc_bow, minimum_probability=0.05)
    
    # return the most pertinent topic :
    if doc_topics:
        doc_topics.sort(key = itemgetter(1), reverse=True)
        theme = topics[doc_topics[0][0]]
        theme_proba = round(doc_topics[0][1], 3)
        if theme == "unknown":
            try :
                theme = topics[doc_topics[1][0]]
                theme_proba = round(doc_topics[1][1], 3)
            except IndexError :
                theme = topics[doc_topics[0][0]]
                theme_proba = round(doc_topics[0][1], 3) 
    else:
        theme = "unknown"
        theme_proba = 0
        
    return pd.Series([theme, theme_proba])


#download data as csv :
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def Theme_extraction(article) :
    return {"Topic" : cluster_similar_documents(article)[0], 
            "Topic proba" : round(cluster_similar_documents(article)[1], 3)}


"""
# Title :
st.title('Topic modeling :')

# Image :
st.image(image, caption='Topic modeling schema')
"""

tab1, tab2= st.tabs(["Dataset clusturing and theme analysis", "Theme extraction"])

def main():
    
    with tab1 :
        st.header("Dataset clusturing and theme analysis")
        # Import corpus CSV :
        uploaded_file = st.file_uploader("Choose a file (only accept .csv)")
        if uploaded_file is not None :
            df_corpus_test = pd.read_csv(uploaded_file)
            # Give column of articles name :
            article_col = st.text_input('Articles column name', 'Article')
            try :
                df_corpus_test[["topic", "topic_proba"]] = df_corpus_test[article_col].apply(lambda corpus : cluster_similar_documents(corpus))
            except KeyError :
                st.write("This column doesn't exist")
            
            # Analyse the result of the clustering
            nb_of_article = pd.DataFrame(df_corpus_test['topic'].value_counts()).rename(columns={"topic": "nb_of_article"})
            st.dataframe(nb_of_article)
            
            #download dat as csv :
            csv = convert_df(nb_of_article)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='topic_analysis.csv',
                mime='text/csv')
        
    with tab2 :
        st.header("Theme extraction from article")
    
        # Predict topic from an article :
        article = st.text_area(label='Wikipedia article to analyze', value="Wikipedia article", height=20)
        if article is not None :
            if st.button('Find topic'):
                st.metric("Topic", Theme_extraction(article)["Topic"])
                st.metric("Topic proba", "{:.3f}".format(Theme_extraction(article)["Topic proba"]))
    

# __name__ :
if __name__ == '__main__' :
    main()   
           
             