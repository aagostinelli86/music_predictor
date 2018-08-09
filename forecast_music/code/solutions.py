
# coding: utf-8

# # A Model for Genre Classification 

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import re
import scipy as sp
import warnings
#%matplotlib inline
import os
import sys


# Loading raw data 

# In[2]:

tracks = "tracks.csv"
sessions = "sessions.csv"
tracks_to_complete = "tracks_to_complete.csv"

df_tracks = pd.read_csv(tracks)
df_sessions = pd.read_csv(sessions)
df_tracks_to_complete = pd.read_csv(tracks_to_complete)


# Before dealing with any model it is worth exploring the dataframes in order to understand the statistics and to perform some data quality checks.

# #### Lyrics data exploration 

# Let's check some properties of the genre distribution

# In[3]:

print df_tracks.shape
df_tracks = df_tracks.drop_duplicates(keep="first")
df_tracks.head(5)


# In[4]:

df_tracks['duration'].describe(include='all')
genres = df_tracks["genre"].unique()


# In[5]:

describeTracks = df_tracks[["genre","duration"]].groupby("genre").describe()
describeTracks 


# It is pretty unuasual to have lyrics with duration of 0 seconds (or 1,2s, ...)! 
# 
# The tracks cataoogue reports several NAs values in the "duration" feature. In order to deal with missing data, the average lyrics duration per genre is computed and used in replacement:   

# In[6]:

blues_avg = describeTracks["duration"]["mean"]["blues"]
electro_avg = describeTracks["duration"]["mean"]["electro"]
rap_avg = describeTracks["duration"]["mean"]["rap"]
raggae_avg = describeTracks["duration"]["mean"]["reggae"]
rock_avg = describeTracks["duration"]["mean"]["rock"]

# fillna 
df_tracks.loc[(df_tracks["duration"].isnull()) & (df_tracks["genre"]=="blues"),"duration"] = blues_avg
df_tracks.loc[(df_tracks["duration"].isnull()) & (df_tracks["genre"]=="electro"),"duration"] = electro_avg
df_tracks.loc[(df_tracks["duration"].isnull()) & (df_tracks["genre"]=="rap"),"duration"] = rap_avg
df_tracks.loc[(df_tracks["duration"].isnull()) & (df_tracks["genre"]=="reggae"),"duration"] = raggae_avg
df_tracks.loc[(df_tracks["duration"].isnull()) & (df_tracks["genre"]=="rock"),"duration"] = rock_avg


# In[7]:

# check 
print "Na values left:\n%s" % df_tracks.isnull().sum() 


# In[8]:

df_tracks["genre"].describe(include='all')


# #### Session data exploration 

# In[9]:

df_sessions.head(5)


# In[10]:

print df_sessions.shape
print "The ratio of NA is\n%s" % str(df_sessions.isnull().sum()/df_sessions.shape[0]*100)


# Since there is less than 1% of NAs in each column, missing observations will be simply filtered out.

# In[11]:

df_sessions = df_sessions.dropna()#.shape() #= df_sessions[]
n_items = len(df_sessions.song_id.unique())
n_users = len(df_sessions.user_id.unique())
print df_sessions.shape
print n_items
print n_users


# There are 2 duplicates rows 

# In[12]:

df_sessions = df_sessions.drop_duplicates(keep="first")
print df_sessions.shape


# Some techniques for recommendation usually take into account a field summarising the preferences given by the users. First of all, we want to count how many times a certain user listened to the same song:

# In[13]:

df_UsrRating = (df_sessions.
               groupby(["song_id","user_id"])
               .count().reset_index().rename(columns={'timestamp':'usr_rating'})
       )
print df_UsrRating.shape
print df_UsrRating["usr_rating"].describe()
df_UsrRating.head(5)


# It is worth noting that this feature is highly unevenly distributed to a value of 1, with tails up to 5.

# In[14]:

plt.hist(df_UsrRating.usr_rating)


# In[15]:

#We could clip the play counts to be binary, e.g., any number greater than 2 is mapped to 1, otherwise it's 0.
#df_UsrRating.loc[df_UsrRating.usr_rating == 1,"usr_rating"] = 0
#df_UsrRating.loc[df_UsrRating.usr_rating > 1,"usr_rating"] = 1


# Another feature to add to the model is the lyrics "popularity", derived looking at how many times a track was globally played. This quantity will be scaled by its maximum measure, in order to get a rating classification between 0 and 1 

# In[16]:

df_GlobRating = (df_sessions[["user_id","song_id"]]
                 .groupby("song_id")
                 .count()
                 .reset_index()
                 .rename(columns={'user_id':'popolarity'}))

df_GlobRating["popolarity"] = df_GlobRating["popolarity"]/df_GlobRating["popolarity"].max()#head(10)


# In[17]:

df_rating = df_GlobRating.merge(df_UsrRating, how = "left", on=["song_id"])


# In[18]:

np.sqrt(np.var(df_rating.usr_rating))


# #### Data manipulation:  merge & filtering

# - We want to aggregte all the data into a single dataframe. 
# 
# - NaNs will be generated where user sessions cannot be mapped to the track catalogues. 
# 
# - The $\textit{song_id}$ corresponding to these NAs will be filtered out in a differnt dataframe, namely $\textit{df_preds}$, that will be used at the vey end for predictions. 
# 
# - Data in $\textit{tracks_to_complete.csv}$ should be at least a subset of $\textit{df_preds}$ 

# In[19]:

df_sessions = df_sessions.merge(df_rating, how = "left", on=["user_id","song_id"])


# In[20]:

dfOverall = df_sessions.merge(df_tracks, how = "left", on=["song_id"])


# In[21]:

print(dfOverall.shape)
dfOverall.sort_values(by=["user_id","song_id","timestamp"]).head(10)


# Each user listenend to a total number of tracks equal to: 

# In[22]:

totNtracks = dfOverall[["user_id","song_id"]].groupby("user_id").count().reset_index().rename(columns={'song_id':'usr_totTrack'})
plt.plot(totNtracks.usr_totTrack)


# In[23]:

print "minumum number of listened tracks %d" % (totNtracks.usr_totTrack.min())


# On average, each user listens to music tracks for the following average duration (in seconds)

# In[24]:

avgTime = dfOverall[["user_id","duration"]].groupby("user_id").mean().reset_index().rename(columns={'duration':'usr_avgSession'})
#avgTime.head(5)


# In[25]:

plt.plot(avgTime.usr_avgSession)


# merging ..

# In[26]:

globFeatures = avgTime.merge(totNtracks,how = "inner", on=["user_id"] )


# In[27]:

dfOverall = dfOverall.merge(globFeatures, how = "left", on=["user_id"])
print(dfOverall.shape)


# In[28]:

df_preds = dfOverall[dfOverall.genre.isnull()]
df = dfOverall[-dfOverall.genre.isnull()]


# In[29]:

print( "Data is split in a DF for modelling with %d observations and in a DF for final predictions with %d " % (df.shape[0],df_preds.shape[0]))


# In[30]:

#df_preds.sort_values(["song_id","usr_rating"])


# In[31]:

df.sort_values(by=["user_id","song_id"], ascending = True).head()


# Let's compare the song_id that will be used for predictions with respect to the merged data

# In[32]:

print "how many sample? %d " % df_tracks_to_complete.song_id.size
print "Exact intersection? %s" % str(set(df_preds.song_id).intersection(set(df_tracks_to_complete.song_id)) == set(df_tracks_to_complete.song_id))
print "how many common sample? %d" % len(set(df_preds.song_id).intersection(set(df_tracks_to_complete.song_id)))
print "was a fully matching subset? %s" % str(set(df_tracks_to_complete.song_id).issubset(set(df_preds.song_id)))


# Therefore, there are some elements in $\textit{df_tracks_to_complete}$ that are not present in $\textit{df_preds}$.
# 
# These elements should be filtered out from $\textit{df_tracks_to_complete}$ since it is not possible to model anything due to the lack of any inforamtion.

# In[33]:

## elememnts in df_tracks_to_complete that are not in df_preds ? 
discrepancy = (set(df_tracks_to_complete.song_id).difference(set(df_preds.song_id)))
print "In tracks_to_complete.csv there are %d observations without any session/catalogue information" % len(discrepancy)
df_tracks_to_complete_filtered = df_tracks_to_complete[~df_tracks_to_complete["song_id"].isin(list(discrepancy))]


# In[34]:

len(df.song_id.unique())


# Up to now, the **cleaned** dataset that will be used for modelling is composed by **101 users**, **5 genres**, **4758 lyrics** and **46084 observations** (sessions)   

# ## Modelling 

# Different techniques should be applied. Standard classifier (e.g., neighbours search and gradient boosting) will be explored. A different solution could arise from the user pattern analysis of lyrics temporal sequences.
# 
# Aiming to achieve quick resultss, I will divide the data only in training (70%) and test (30%) sets, without performing cross validation for hyper-parameters tuning, as a first approximation to find out the best model. 

# In[35]:

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


# In[36]:

from __future__ import print_function

import logging
import numpy as np
from time import time

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.metrics import classification_report


# In[37]:

def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    

    print("classification report:")
    print(metrics.classification_report(y_test, pred))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


# #### Data pre-processing 1

# Since we are interested in predicting only the genre, I will drop the "duration" information that is missing when doing the predictions.

# In[38]:

dfWhole = df.drop(labels=["duration"], axis=1).set_index("timestamp")


# In[39]:

dfWhole.head()


# Finally, let's create a train and test dataset by using stratified sampling, to reproduce the relative unbalanced proportions of music genre in the session datasets. Indeeds, some genre frequencies were unbalance already at the level of the track dataset:

# In[40]:

df_tracks.groupby("genre").count()["song_id"]


# In[41]:

X_train, X_test, y_train, y_test = train_test_split(dfWhole.drop(labels=['genre'], axis=1), dfWhole.genre, test_size = 0.3, stratify=dfWhole.genre,random_state=42)


# #### Models

# Several classification approaces are explored. Togheter with the accuracy metrics, a more suitable performance metrics is the f1_score, since genres classes are not balanced.

# I will concentrate more on a kNN classifier. To address the inefficiencies of KD Trees in higher dimensions, the ball tree data structure will be used as algorithms methods. 

# In[42]:

results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (KNeighborsClassifier(n_neighbors=20, algorithm = "ball_tree"), "kNN_20_ball"),
        #(KNeighborsClassifier(n_neighbors=10, algorithm = "ball_tree"), "kNN_10_ball"),
        (KNeighborsClassifier(n_neighbors=50, algorithm = "ball_tree"), "kNN_50_ball"),
        #(KNeighborsClassifier(n_neighbors=5, algorithm = "ball_tree"), "kNN_5_ball"),
        (KNeighborsClassifier(n_neighbors=100, algorithm = "ball_tree"), "kNN_100_ball"),
        #(KNeighborsClassifier(n_neighbors=20, metric="cosine", algorithm = "brute"), "kNN_1"),
        (KNeighborsClassifier(n_neighbors=50, metric="cosine",algorithm = "brute"), "kNN_50"),
        #(KNeighborsClassifier(n_neighbors=200, metric="cosine",algorithm = "brute"), "kNN_200"),
        (AdaBoostClassifier( n_estimators=50), "adaboost_50"),
        (AdaBoostClassifier( n_estimators=500), "adaboost_500"),
        (GradientBoostingClassifier(n_estimators=2000), "gradientboost_2000"),
        (GradientBoostingClassifier(n_estimators=3000), "gradientboost_3000"),
        #(GradientBoostingClassifier(n_estimators=900), "gradientboost_900"),
        #(GradientBoostingClassifier(n_estimators=1200), "gradientboost_1200"),
        #(GradientBoostingClassifier(n_estimators=1000), "gradientboost_1000"),
        #(GradientBoostingClassifier(n_estimators=1500), "gradientboost_1500"),
        (RandomForestClassifier(n_estimators=500), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))


# In[43]:

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))


# In[44]:

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))


# In[45]:

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()


# ### Best Hyperparameters Grdid Search with Cross Validation

# To achieve better results with respect to those already obtained, a fine-tuning of the hyper-parameters by mean of a gridSearch cross validation shoudl be expected. It is highly computational requirinq, though.

# ## Running the best model on the unobserved dataset

# The **best preformance** has been achieved by using a **GradientBoostingClassifier** with **3000 estimators** (decision trees). This model has performed with an **accuracy of 91.2%** and an average **f1_score of 91% ** on the test set.

# The structure of the training and test sets is recreated: 

# In[42]:

df_preds = df_preds.drop(labels=["duration","genre"], axis=1).set_index("timestamp")


# In[43]:

bestModel = GradientBoostingClassifier(n_estimators=3000)


# In[44]:

bestModel.fit(dfWhole.drop(labels="genre", axis=1), dfWhole.genre)


# In[45]:

import pickle
pickle.dump( bestModel, open( "BestModel.p", "wb" ) )


# Finally, prediction of the music genre are generetaed and printed out on a csv file, after having filtered for uninformative tracks with any information in the session files. 

# In[46]:

pred = bestModel.predict(df_preds)


# In[47]:

df_preds["predicted_genre"] = pred


# In[48]:

df_preds = df_preds.reset_index()


# In[49]:

#df_preds[["user_id","song_id","timestamp"]].sort_values(["user_id","song_id","timestamp"])


# In[144]:

df_preds[["song_id","predicted_genre"]].to_csv("tmp_overall_predictions.csv",index =False)


# In[51]:

filteredLis = (list(set(df_preds.song_id).intersection(set(df_tracks_to_complete.song_id))))


# For a given song_id there might be multiple answers due to: 
# 
# - same genre predictions from different user_id
# - different genre predictions from different user_id
# 
# In the first case, a filtering across replicates will be usesd, whereas for the second case for simplicity the first answer among the possible outcomes will be picked up.   

# In[70]:

df_predsFiltered = df_preds[["song_id","predicted_genre"]].sort_values(["song_id","predicted_genre"]).drop_duplicates(keep="first")


# In[140]:

df_Final = df_predsFiltered.drop_duplicates(subset="song_id", keep="first")
print(df_Final.shape)


# This was a quick and dirty work-around. In order to map the outcome best estimation with the genre measured distribution, the best choice would be done by extracting a random number $f_r$ from a uniform distribution in [0,1] and comparing its value with the per class frequecnies, given by:
# 

# In[136]:

df_predsFiltered[df_predsFiltered.song_id.isin(filteredLis)].shape


# In[142]:

df.groupby("genre").count()["song_id"]/df.shape[0]


# So that:
#   - if $fgenre_j < fr < fgenre_i$  --> Genre "i" is chosen 
#   - if $f_r < fgenre_k< fgenre_i < fgenre_j$ --> Genre "k" is chosen 
#   - ad so on ..
#   

# We can finally save the list of predictions in the format requested by the test

# In[143]:

df_predsFiltered.to_csv("solutions.csv",index =False)


# # Appendix 

# By using only the information given by song_id, user_id and the number of times a lyrics was played (proxy of the rating), it is possible to perform a pivot of the previous dataframes in order to create a Df with lyrics labels associated to a given combination of user. This methodology is closer to what is usually done in collaborative filtering but it is not as performing as the previous approach

# #### Data pre-processing 2

# In[64]:

df_filtered = df.drop(labels=["timestamp"], axis=1).drop_duplicates(keep="first")


# Pivoting

# In[65]:

df_pivot = df_filtered.pivot(index = 'song_id', columns ='user_id', values = 'usr_rating').fillna(0).reset_index()


# In[66]:

reshaped_df = (df_pivot
               .merge(df_filtered.drop(labels=["user_id","usr_rating","usr_avgSession","duration","usr_totTrack","popolarity"], axis=1), how = "inner", on=["song_id"])
               .drop_duplicates(keep = "first"))


# It is possible now to create a train and test dataset by using stratified sampling, to respect the relative proportions of music genres in the session datasets. Indees, some genre frequencies were unbalance already at the level of the track dataset

# In[67]:

df_tracks.groupby("genre").count()["song_id"]


# In[68]:

X_train, X_test, y_train, y_test = train_test_split(reshaped_df.drop(labels=['genre'], axis=1), reshaped_df.genre, test_size = 0.3, stratify=reshaped_df.genre,random_state=42)


# In[69]:

X_train.head()


# kf = KFold(n_splits=5)
# for train, test in kf.split(X):
#     print("%s %s" % (train, test))
#     

# In[70]:

results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (KNeighborsClassifier(n_neighbors=20, algorithm = "ball_tree"), "kNN_20_ball"),
        (KNeighborsClassifier(n_neighbors=1, metric="cosine", algorithm = "brute"), "kNN_1"),
        (KNeighborsClassifier(n_neighbors=4, metric="cosine",algorithm = "brute"), "kNN_4"),
        (KNeighborsClassifier(n_neighbors=10, metric="cosine",algorithm = "brute"), "kNN_10"),
        (KNeighborsClassifier(n_neighbors=50, metric="cosine",algorithm = "brute"), "kNN_50"),
        (KNeighborsClassifier(n_neighbors=200, metric="cosine",algorithm = "brute"), "kNN_200"),
        (AdaBoostClassifier( n_estimators=50), "adaboost_50"),
        (AdaBoostClassifier( n_estimators=500), "adaboost_500"),
        (GradientBoostingClassifier(n_estimators=50), "gradientboost_50"),
        (GradientBoostingClassifier(n_estimators=500), "gradientboost_500"),
        (RandomForestClassifier(n_estimators=500), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))


# In[71]:

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))


# In[72]:

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))


# In[73]:

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()

