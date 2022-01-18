import re
import utils
import pickle
import dvc.api
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

    with open("./version.txt", mode = "w") as f:
        f.write(f"{len(dvc_train)}, {len(dvc_test)}")

    return df

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

def main():
    df = load_data()
    train, test = data_preproc(df)
    bow_train, bow_test = bag_of_words(train, test)

    col = get_bow_columns(bow_train)

    model = model_training(bow_train[col], train)

    pickle.dump(model, open("../model/model.pkl", 'wb'))

    logregpred = model.predict_proba(bow_test[col])
    pred_logreg = []
    for i in range(0,len(logregpred)):
        pred_logreg.append(utils.argmax_2(logregpred[i]))


    acc_score = round(accuracy_score(test['sentiment'].tolist(),pred_logreg), 5)
    print("accuracy: ", acc_score)
    if acc_score <= 0.4:
        raise ValueError("Accuracy is too low")
    elif acc_score >= 0.95:
        raise ValueError("Accuracy is too high")
    else:
        pass

if __name__ == "__main__":
    main()
