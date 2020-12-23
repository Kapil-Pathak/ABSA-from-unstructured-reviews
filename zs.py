### References: https://github.com/ishikaarora/Aspect-Sentiment-Analysis-on-Amazon-Reviews
import pandas as pd
from sklearn import cluster
from collections import defaultdict
import spacy, json
import numpy as np
import pandas as pd
import os,nltk
import pickle

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
"""Create a list of common words to remove"""
stop_words=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
            "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
            "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
            "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
            "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
            "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

exclude_stopwords = ['it','this','they','these']
prod_pronouns = ['it','this','they','these']
NUM_CLUSTERS = 10

def extract_aspects(reviews,nlp,sid):
    print("Entering Apply function!")
    aspect_list = reviews.apply(lambda row: apply_extraction(row,nlp,sid)) #going through all the rows in the dataframe
    return aspect_list

def init_spacy():
    print("\nLoading spaCy Model....")
    nlp = spacy.load('en_core_web_lg')
    print("spaCy successfully loaded")
    for w in stop_words:
        nlp.vocab[w].is_stop = True
    for w in exclude_stopwords:
        nlp.vocab[w].is_stop = False
    return nlp

def init_nltk():
    print("\nLoading NLTK....")
    try :
        sid = SentimentIntensityAnalyzer()
    except LookupError:
        print("Please install SentimentAnalyzer using : nltk.download('vader_lexicon')")
    print("NLTK successfully loaded")
    return(sid)


def apply_extraction(row,nlp,sid):
    sent_list = nltk.tokenize.sent_tokenize(row)
    rule1_pairs = []
    rule2_pairs = []
    rule3_pairs = []
    rule4_pairs = []
    rule5_pairs = []
    rule6_pairs = []
    rule7_pairs = []
    for sent in sent_list:

        doc=nlp(sent.lower())
        #print(doc)

        ## FIRST RULE OF DEPENDANCY PARSE -
        ## M - Sentiment modifier || A - Aspect
        ## RULE = M is child of A with a relationshio of amod

        for token in doc:
            A = "999999"
            M = "999999"
            if token.dep_ == "amod" and not token.is_stop:
                M = token.text
                A = token.head.text
                #print(A)
                # add adverbial modifier of adjective (e.g. 'most comfortable headphones')
                M_children = token.children
                for child_m in M_children:
                    if(child_m.dep_ == "advmod"):
                        M_hash = child_m.text
                        M = M_hash + " " + M
                        #print(1, A,  M)


                # negation in adjective, the "no" keyword is a 'det' of the noun (e.g. no interesting characters)
                A_children = token.head.children
                #print("###############")
                for child_a in A_children:
                    if(child_a.dep_ == "det" and child_a.text == 'no'):
                        neg_prefix = 'not'
                        M = neg_prefix + " " + M
                        #print(1, A, M)


            if(A != "999999" and M != "999999"):
                print(1, M, A)
                rule1_pairs.append((A, M,sid.polarity_scores(token.text)['compound'],1))

        ## SECOND RULE OF DEPENDANCY PARSE -
        ## M - Sentiment modifier || A - Aspect
        #Direct Object - A is a child of something with relationship of nsubj, while
        # M is a child of the same something with relationship of dobj
        #Assumption - A verb will have only one NSUBJ and DOBJ

        for token in doc:
            children = token.children
            A = "999999"
            M = "999999"
            add_neg_pfx = False
            for child in children :
                if(child.dep_ == "nsubj" and not child.is_stop):
                    A = child.text
                    # check_spelling(child.text)

                if((child.dep_ == "dobj" and child.pos_ == "ADJ") and not child.is_stop):
                    M = child.text
                    #check_spelling(child.text)

                if(child.dep_ == "neg"):
                    neg_prefix = child.text
                    add_neg_pfx = True

        if (add_neg_pfx and M != "999999"):
            M = neg_prefix + " " + M
            print(2, M, A)
            if(A != "999999" and M != "999999"):
                rule2_pairs.append((A, M,sid.polarity_scores(M)['compound'],2))


        ## THIRD RULE OF DEPENDANCY PARSE -
        ## M - Sentiment modifier || A - Aspect
        ## Adjectival Complement - A is a child of something with relationship of nsubj, while
        ## M is a child of the same something with relationship of acomp
        ## Assumption - A verb will have only one NSUBJ and DOBJ
        ## "The sound of the speakers would be better. The sound of the speakers could be better" - handled using AUX dependency



        for token in doc:

            children = token.children
            A = "999999"
            M = "999999"
            add_neg_pfx = False
            for child in children :
                if(child.dep_ == "nsubj" and not child.is_stop):
                    A = child.text
                    # check_spelling(child.text)

                if(child.dep_ == "acomp" and not child.is_stop):
                    M = child.text

                # example - 'this could have been better' -> (this, not better)
                # MD: A modal verb is a type of verb that is used to indicate modality,
                # that is: likelihood, ability, permission,
                if(child.dep_ == "aux" and child.tag_ == "MD"):
                    neg_prefix = "not"
                    add_neg_pfx = True

                if(child.dep_ == "neg"):
                    neg_prefix = child.text
                    add_neg_pfx = True

            if (add_neg_pfx and M != "999999"):
                M = neg_prefix + " " + M
                    #check_spelling(child.text)

            if(A != "999999" and M != "999999"):
                print(3, M, A)
                rule3_pairs.append((A, M, sid.polarity_scores(M)['compound'],3))

        ## FOURTH RULE OF DEPENDANCY PARSE -
        ## M - Sentiment modifier || A - Aspect

        #Adverbial modifier to a passive verb - A is a child of something with relationship of nsubjpass, while
        # M is a child of the same something with relationship of advmod

        #Assumption - A verb will have only one NSUBJ and DOBJ

        for token in doc:


            children = token.children
            A = "999999"
            M = "999999"
            add_neg_pfx = False
            for child in children :
                if((child.dep_ == "nsubjpass" or child.dep_ == "nsubj") and not child.is_stop):
                    A = child.text
                    # check_spelling(child.text)

                if(child.dep_ == "advmod" and not child.is_stop):
                    M = child.text
                    M_children = child.children
                    for child_m in M_children:
                        if(child_m.dep_ == "advmod"):
                            M_hash = child_m.text
                            M = M_hash + " " + child.text
                            break
                    #check_spelling(child.text)

                if(child.dep_ == "neg"):
                    neg_prefix = child.text
                    add_neg_pfx = True

            if (add_neg_pfx and M != "999999"):
                M = neg_prefix + " " + M

            if(A != "999999" and M != "999999"):
                print(4, M, A)
                rule4_pairs.append((A, M,sid.polarity_scores(M)['compound'],4)) # )


        ## FIFTH RULE OF DEPENDANCY PARSE -
        ## M - Sentiment modifier || A - Aspect

        #Complement of a copular verb - A is a child of M with relationship of nsubj, while
        # M has a child with relationship of cop

        #Assumption - A verb will have only one NSUBJ and DOBJ

        for token in doc:
            children = token.children
            A = "999999"
            buf_var = "999999"
            for child in children :
                if(child.dep_ == "nsubj" and not child.is_stop):
                    A = child.text
                    # check_spelling(child.text)

                if(child.dep_ == "cop" and not child.is_stop):
                    buf_var = child.text
                    #check_spelling(child.text)

            if(A != "999999" and buf_var != "999999"):
                print(5, M, A)
                rule5_pairs.append((A, token.text,sid.polarity_scores(token.text)['compound'],5))


        ## SIXTH RULE OF DEPENDANCY PARSE -
        ## M - Sentiment modifier || A - Aspect
        ## Example - "It ok", "ok" is INTJ (interjections like bravo, great etc)


        for token in doc:
            children = token.children
            A = "999999"
            M = "999999"
            if(token.pos_ == "INTJ" and not token.is_stop):
                for child in children :
                    if(child.dep_ == "nsubj" and not child.is_stop):
                        A = child.text
                        M = token.text
                        # check_spelling(child.text)

            if(A != "999999" and M != "999999"):
                print(6, M, A)
                rule6_pairs.append((A, M,sid.polarity_scores(M)['compound'],6))



    aspects = []
    aspects = rule1_pairs +rule2_pairs+rule3_pairs+rule4_pairs+rule5_pairs+rule6_pairs+rule7_pairs

    # replace all instances of "it", "this" and "they" with "product"
    prod_pronouns = ['it','this','they','these']
    aspects = [(A.strip() ,M,P,r) if A not in prod_pronouns else ("product",M,P,r) for A,M,P,r in aspects ]
    return aspects
#doc=nlp(df['content'])
def get_aspect_freq_map(aspects):
    aspect_freq_map = defaultdict(int)
    for asp in aspects:
        aspect_freq_map[asp] += 1
    return aspect_freq_map

def get_unique_aspects(aspects):
    unique_aspects = list(set(aspects)) # use this list for clustering
    return unique_aspects
# @profile
def get_word_vectors(unique_aspects, nlp):
    asp_vectors = []
    for aspect in unique_aspects:
        # print(aspect)
        token = nlp(aspect)
        asp_vectors.append(token.vector)
    return asp_vectors


def get_word_clusters(unique_aspects, nlp):
    asp_vectors = get_word_vectors(unique_aspects, nlp)
    if len(unique_aspects) <= NUM_CLUSTERS:
        return list(range(len(unique_aspects)))
    n_clusters = NUM_CLUSTERS
    model = cluster.KMeans(n_clusters=n_clusters)
    model.fit(asp_vectors)
    labels = model.labels_
    return labels, model

def get_cluster_names_map(asp_to_cluster_map, aspect_freq_map):
    cluster_id_to_name_map = defaultdict()
    clusters = set(asp_to_cluster_map.values())
    for i in clusters:
        this_cluster_asp = [k for k,v in asp_to_cluster_map.items() if v == i]
        filt_freq_map = {k:v for k,v in aspect_freq_map.items() if k in this_cluster_asp}
        filt_freq_map = sorted(filt_freq_map.items(), key = lambda x: x[1], reverse = True)
        cluster_id_to_name_map[i] = filt_freq_map
    return cluster_id_to_name_map

if __name__ == '__main__' :
    nlp = init_spacy()
    sid = init_nltk()
    df = pd.read_csv('amazon_sample_reviews.csv')
    print("----------------***----------------")
    print("\nExtracting aspect pairs")
    aspects = extract_aspects(df['content'],nlp,sid)
    total_aspects = []
    for aspect, review in zip(aspects, df['content']):
        total_aspects.extend([a[0] for a in aspect])
        #1963 0 653 241 0 0 185

    unique_aspects = get_unique_aspects(total_aspects)
    aspect_freq_map = get_aspect_freq_map(total_aspects)
    aspect_labels, kmeans = get_word_clusters(unique_aspects, nlp)
    asp_to_cluster_map = dict(zip(unique_aspects, aspect_labels))
    cluster_names_map = get_cluster_names_map(asp_to_cluster_map, aspect_freq_map)


    with open('data.p', 'wb') as fp:
        pickle.dump(cluster_names_map, fp, protocol=pickle.HIGHEST_PROTOCOL)
    filename = 'finalized_model.sav'
    pickle.dump(kmeans, open(filename, 'wb'))
    loaded_model = pickle.load(open(filename, 'rb'))
    for i in list(cluster_names_map.keys()):
        print(list(cluster_names_map[i]))
        print(list(cluster_names_map[i][0]))
        print(cluster_names_map[i][0][0])
    print(kmeans.predict([nlp("Mattress").vector]))
    for item in df['content']:
        exe_aspects = apply_extraction(item,nlp,sid)
        sentiment_scores = [0]*NUM_CLUSTERS
        aspects_found = [False]*NUM_CLUSTERS
        for exe in exe_aspects:
            sentiment_scores[kmeans.predict([nlp(exe[0]).vector])[0]] += exe[2]
            aspects_found[kmeans.predict([nlp(exe[0]).vector])[0]] = True
        #print(item)
        #print(sid.polarity_scores(item))
        for i in range(len(sentiment_scores)):
            if aspects_found[i]:
                if sentiment_scores[i]>0:
                    #print(cluster_names_map[i][0][0], "Positive")
                    pass
                elif sentiment_scores[i]<0:
                    #print(cluster_names_map[i][0][0], "Negative")
                    pass
                else:
                    #print(cluster_names_map[i][0][0], "Neutral")
                    pass
