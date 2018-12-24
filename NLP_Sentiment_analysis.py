#%% all imports
import pandas as pd
import numpy as np
import glob
from sklearn.utils import shuffle

from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import string
from nltk import FreqDist
from nltk import NaiveBayesClassifier
from nltk import classify
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
#%%creating a csv file for all the reviews
'''
###### for positive reviews
path_pos = r'/home/nilesh/nltk_data/corpora/movie_reviews/pos/'

list_of_file_names = glob.glob1(path_pos,'*') 
list_of_file_names = sorted(list_of_file_names)
len(list_of_file_names)

pre_df = []
for i in range(len(list_of_file_names)):
    file = open(path_pos+list_of_file_names[i],'r')
    pre_df.append(file.read())

df = pd.DataFrame(pre_df)
df['review_senti'] = 1
df.columns = ['review_text','review_senti']

df.to_csv(path_pos+'all_combine_pos.csv',index=None)
'''
#%%
'''
#### for negative reviews
del df,pre_df
path_neg = r'/home/nilesh/nltk_data/corpora/movie_reviews/neg/'

list_of_file_names = glob.glob1(path_neg,'*') 
list_of_file_names = sorted(list_of_file_names)
len(list_of_file_names)

pre_df = []
for i in range(len(list_of_file_names)):
    file = open(path_neg+list_of_file_names[i],'r')
    pre_df.append(file.read())

df = pd.DataFrame(pre_df)
df['review_senti'] = 0
df.columns = ['review_text','review_senti']

df.to_csv(path_neg+'all_combine_neg.csv',index=None)
'''
#%%
'''
#### combining the two
path = r'/home/nilesh/nltk_data/corpora/movie_reviews/'

df_pos = pd.read_csv(path+'all_combine_pos.csv')
df_neg = pd.read_csv(path+'all_combine_neg.csv')

df_all_original_review = pd.concat([df_pos,df_neg])

df_all_original_review.to_csv(path+'all_reviews_pos_neg.csv',index=None)

'''
#%%fetching the data from excel file
path = r'/home/nilesh/nltk_data/corpora/movie_reviews/all_reviews_pos_neg.csv'

df_original_reviews_posNneg = pd.read_csv(path)

#shuffling the data
df_original_reviews_posNneg = shuffle(df_original_reviews_posNneg)

#%%
#getting the list of all words from both the documents - Vocab
document_list = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        document_list.append((movie_reviews.words(fileid),category))

#extracting all the words from all the documents
all_words_list = [word.lower() for word in movie_reviews.words()]
#its a very large number 1583820 

#sorting the list
all_words_list = sorted(all_words_list)
#removing duplicates
all_words_list = list(set(all_words_list))
#creating stopwords list and punctions list
stop_words = list(stopwords.words('english'))
punc_list = list(string.punctuation)

#initializing the lemmetizer
w_l = WordNetLemmatizer()
#getting the clean list of words
all_words_list_cleaned = []
for word in all_words_list:
    if len(word)>2 and word not in stop_words and word not in punc_list:
        all_words_list_cleaned.append(w_l.lemmatize(word))
        
#lemmitization might have bougt in some duplicate words so removing them again
all_words_list_cleaned = list(set(all_words_list_cleaned))
vocab = sorted(all_words_list_cleaned)

#creating a dataframe that is our feature vector
df = pd.DataFrame(vocab)
#df.to_csv('/home/nilesh/2. Nilesh Learnings 15thNOV18 onwards/02. All dot_py files/Outputfile/program5_vocab.csv',index=None)     
df = df.iloc[382:,:] #after manually checking it was found that till 383 rows still had some combination of punctuation characters; so removing it.
df = df.transpose()
df.columns = df.iloc[0,:]
df = df.iloc[1:,:]

#%%fill the vocab with appropriate positive scores

df_vocab = pd.DataFrame(vocab)
df_vocab = df_vocab.iloc[382:,:] #after manually checking it was found that till 383 rows still had some combination of punctuation characters; so removing it.
df_vocab['pos-neg-score'] = 0
df_vocab.columns = ['actual_words','pos-neg-score']

'''
pos_score = 0
neg_score = 0
for w in list(df_vocab.iloc[:,0]):
    senti_synsets = swn.senti_synsets(w.lower())
    for senti_synset in senti_synsets:
        p = senti_synset.pos_score()
        n = senti_synset.neg_score()
        pos_score+=p
        neg_score+=n
'''
for i in range(len(list(df_vocab.iloc[:,0]))):
    pos_score = 0
    neg_score = 0
    senti_synsets = swn.senti_synsets(df_vocab.iloc[i,0])
    for senti_synset in senti_synsets:
        p = senti_synset.pos_score()
        n = senti_synset.neg_score()
        pos_score+=p
        neg_score+=n
    df_vocab.iloc[i,1] = pos_score - neg_score

#in further processing we are going to use zero 0 as no value for the algorithm means the word did not appear in that sentence so the scores
#that we have got as zero should be give a very small but NON ZERO value.
    
#its also observed after a few trials that 34k plus features are not supported by a single machine so keeping only those features that had significant values

df_vocab = df_vocab[df_vocab.iloc[:,1] != 0]
len(df_vocab) # lenght Null'
#df_vocab.iloc[i,1] = 'Nan'


df_vocab_transpose = df_vocab.transpose()
df_vocab_transpose.to_csv(r'C:\Users\Sam\Desktop',index=None,header=None)
del df_vocab_transpose

#df_feature_vector = pd.read_csv(r'/home/nilesh/2. Nilesh Learnings 15thNOV18 onwards/02. All dot_py files/Outputfile/program5_vocab.csv')
#initializing all the features with these values.
for i in range(2000):
    df_feature_vector = pd.concat([df_feature_vector,df_feature_vector])
    #df_feature_vector = df_feature_vector.append(df11)#append doesn't happen inplace so we have to assign it again.
    


#%%till this point we have 2 dataframes 
#1.df_all_original_review1 that contains all the review text with its corrosponding reviews
#2.df dataframe that has vocab as features

# in this section we will futher build the df dataframe in such as way as to each row will represent a review and the cross-section of each
# row and each feature will contain only the positive score of that word given by sentinet

df_all_original_review2 = df_all_original_review1.copy()


