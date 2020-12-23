import streamlit as st
import numpy as np
import pickle
from zs import *

def main():
  """ Unstructured Reviews: Aspect-Based Sentiment Analysis """
  #st.image('download.png', width=350)
  # Title
  st.title("Unstructured Reviews: Aspect-Based Sentiment Analysis")
  st.subheader("Rule-Based Aspect Extraction and Sentiment Analysis")
  st.subheader("""
      What are you interested in?
      """)

  # AnswerGeneration
  #context =  st.text_area("Type your context here")
  if st.checkbox("Show Aspect Clusters"):
      st.write("Category: Mattress")
      st.image('matress.png', width=350)
      st.write("Category: product")
      st.image('product.png', width=350)
      st.write("Category: bed")
      st.image('bed.png', width=350)
      st.write("Category: foam")
      st.image('foam.png', width=350)
      st.write("Category: firm")
      st.image('firm.png', width=350)
      st.write("Category: smell")
      st.image('smell.png', width=350)
      st.write("Category: time")
      st.image('time.png', width=350)
      st.write("Category: One")
      st.image('one.png', width=350)
      st.write("Category: daughter")
      st.image('daughter.png', width=350)
      #st.write("Category: xl")
      #st.image('xl.png', width=350)
  if st.checkbox("Aspect Extraction"):
    nlp = init_spacy()
    sid = init_nltk()
    filename = 'finalized_model.sav'
    kmeans = pickle.load(open(filename, 'rb'))
    with open('data.p', 'rb') as fp:
        cluster_names_map = pickle.load(fp)

    st.subheader("Extract Aspects from text")

    #context =  st.text_area("Type your context here")
    item= st.text_input('Review: ', None)
    sent_list = nltk.tokenize.sent_tokenize(item)
    if st.button("Process the review"):
        for i, sent in enumerate(sent_list):
            st.success("Sentence {}: {}".format(i+1, sent))
            exe_aspects = apply_extraction(sent,nlp,sid)
            #if st.button("Show raw aspects-opinion words"):
            if len(exe_aspects)>0:
                st.success("Aspect terms with opinion words: {}\n".format(exe_aspects))
            sentiment_scores = [0]*NUM_CLUSTERS
            aspects_found = [False]*NUM_CLUSTERS
            for exe in exe_aspects:
                sentiment_scores[kmeans.predict([nlp(exe[0]).vector])[0]] += exe[2]
                aspects_found[kmeans.predict([nlp(exe[0]).vector])[0]] = True
            #print(item)
            st.success("Overall Sentiment from a sentence: {}".format(sid.polarity_scores(item)['compound']))
            st.success("Different Categories recorded: ")
            if sum(sentiment_scores) == 0:
                st.success("Category: Product")
                if sid.polarity_scores(item)['compound']>0:
                    st.success("Positive Sentiment with Sentiment Score {}".format(sid.polarity_scores(item)['compound']))
                elif sid.polarity_scores(item)['compound']<0:
                    st.success("Negative Sentiment with Sentiment Score {}".format(sid.polarity_scores(item)['compound']))
                else:
                    st.success("Neutral Sentiment with Sentiment Score 0")
            for i in range(len(sentiment_scores)):
                if aspects_found[i]:
                    if sentiment_scores[i]>0 and cluster_names_map[i][0][0] == 'pain':
                        st.success("Category: {}".format(cluster_names_map[i][0][0]))
                        st.success("Negative Sentiment with Sentiment Score {}".format(-sentiment_scores[i]))
                    elif sentiment_scores[i]>0:
                        st.success("Category: {}".format(cluster_names_map[i][0][0]))
                        st.success("Positive Sentiment with Sentiment Score {}".format(sentiment_scores[i]))
                        #st.success("Positive Sentiment")
                        #pass
                    elif sentiment_scores[i]<0 :
                        st.success("Category: {}".format(cluster_names_map[i][0][0]))
                        st.success("Negative Sentiment with Sentiment Score {}".format(sentiment_scores[i]))
                    else:
                        st.success("Category: {}".format(cluster_names_map[i][0][0]))
                        st.success("Neutral Sentiment with Sentiment Score {}".format(sentiment_scores[i]))
                    #pass
  st.sidebar.subheader("Aspect-Based Sentiment Analysis")
  st.sidebar.text("From Unstructured Reviews ...")
  st.sidebar.info("Demo App")
  st.sidebar.text("By Kapil Pathak")

if __name__ == '__main__':
  main()
