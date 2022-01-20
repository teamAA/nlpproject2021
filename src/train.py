import sys
import re
import utils
import dvc.api
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scikitplot.metrics import plot_roc, plot_confusion_matrix, plot_precision_recall
import neptune
from neptunecontrib.monitoring.lightgbm import neptune_monitor
import matplotlib.pyplot as plt
import os

# firli was here

def load_data():
    dvc_train = dvc.api.read('data-registry/train.csv', repo='../')
    dvc_test = dvc.api.read('data-registry/test.csv', repo='../')

    train = pd.read_csv( StringIO(dvc_train) )
    print(f"train shape: {train.shape}")
    train['dataset'] = 'train'

    test = pd.read_csv( StringIO(dvc_test) )
    test['dataset'] = 'test'

    df = pd.concat([train,test],axis=0)
    df = df.reset_index().drop('index',axis=1)

    return df, len(dvc_train), len(dvc_test)

def data_preproc(df):
    df['sentiment'] = np.where(df['sentiment']=='neutral',0,df['sentiment'])
    df['sentiment'] = np.where(df['sentiment']=='positive',1,df['sentiment'])
    df['sentiment'] = np.where(df['sentiment']=='negative',-1,df['sentiment'])

    df['text'] = df['text'].astype(str)
    # Regex to get letter only 
    df['text'] = [re.sub(r"(@[A-Za-z]+)|([^A-Za-z \t])|(\w+:\/\/\S+)", " ", df['text'][x]) for x in range(0,df.shape[0])]

    df['cleansed_text'] = df['text']\
        .apply(lambda x: utils.tokenization(x.lower()))\
        .apply(lambda x: utils.remove_stopwords(x))\
        .apply(lambda x: utils.lemmatizing(x))\
        .apply(lambda x: utils.stemming(x))\
        .apply(lambda x: utils.empty_token(x))    


    train_cleaned = df[df['dataset']=='train']
    test_cleaned = df[df['dataset']=='test'].reset_index().drop('index',axis=1)

    return train_cleaned, test_cleaned

def bag_of_words(train, test):
    # The default token pattern removes tokens of a single character. That's why we don't have the "I" and "s" tokens in the output
    sentences_train = [' '.join(x) for x in train['cleansed_text']]
    sentences_test = [' '.join(x) for x in test['cleansed_text']]

    count_vectorizer = CountVectorizer()
    count_vectorizer.fit(sentences_train)
    bag_of_words_train = count_vectorizer.transform(sentences_train)
    bag_of_words_test = count_vectorizer.transform(sentences_test)

    # Show the Bag-of-Words Model as a pandas DataFrame
    feature_names = count_vectorizer.get_feature_names()
    df_bow_train = pd.DataFrame(bag_of_words_train.toarray(), columns = feature_names)
    df_bow_test = pd.DataFrame(bag_of_words_test.toarray(), columns = feature_names)

    return df_bow_train, df_bow_test

def model_training(bow_train, train):
    logreg = LogisticRegression(max_iter=1000, random_state=2021)
    logreg.fit(bow_train, train['sentiment'].tolist())
    return logreg

def get_bow_columns(df_bow):
    col = []
    for i in list(df_bow.columns):
        if len(set(i)) >= 3:
            col.append(i)
    return col

def log_neptune(test, pred):
    #Connect your script to Neptune
    neptune.init(project_qualified_name=os.getenv('NEPTUNE_PROJECT_NAME'), api_token=os.getenv('NEPTUNE_API_TOKEN'))
    
    #Create an experiment and log hyperparameters
    neptune.create_experiment('NLP_test')

    f1 = f1_score(test['sentiment'].tolist(), pred, average='macro')
    accuracy = accuracy_score(test['sentiment'].tolist(), pred)
    
    # Log metrics to Neptune
    neptune.log_metric('accuracy', accuracy)
    neptune.log_metric('f1_score', f1)

    #     fig_roc, ax = plt.subplots(figsize=(12, 10))
    #     plot_roc(test['sentiment'].tolist(), pred_logreg, ax=ax)

    #     fig_cm, ax = plt.subplots(figsize=(12, 10))
    #     plot_confusion_matrix(test['sentiment'].tolist(), pred_logreg, ax=ax)

    #     fig_pr, ax = plt.subplots(figsize=(12, 10))
    #     plot_precision_recall(test['sentiment'].tolist(), pred_logreg, ax=ax)

    #     # Log performance charts to Neptune
    #     neptune.log_image('performance charts', fig_roc)
    #     neptune.log_image('performance charts', fig_cm)
    #     neptune.log_image('performance charts', fig_pr)

    #     # Handle CI pipeline details
    #     if os.getenv('CI') == "true":
    #         neptune.append_tag('ci-pipeline', os.getenv('NEPTUNE_EXPERIMENT_TAG_ID'))

def main(log = False):
    df, len_train, len_test = load_data()
    train, test = data_preproc(df)
    bow_train, bow_test = bag_of_words(train, test)

    col = get_bow_columns(bow_train)
    
    model = model_training(bow_train[col], train)

    logregpred = model.predict_proba(bow_test[col])
    pred_logreg = []
    for i in range(0,len(logregpred)):
        pred_logreg.append(utils.argmax_2(logregpred[i]))

    # Testing
    acc_score = round(accuracy_score(test['sentiment'].tolist(),pred_logreg), 5)
    print("accuracy: ", acc_score)
    if acc_score <= 0.4:
        raise ValueError("Accuracy is too low")
    elif acc_score >= 0.95:
        raise ValueError("Accuracy is too high")
    else:
        pass
    
    if log == True:
        log_neptune(test, pred_logreg)
    
    return model
    
if __name__ == "__main__":
    main(log = False)

    sys.exit(0)
