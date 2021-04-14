#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an n-gram baseline for the PAN20 shared task on celebrity profiling.
For usage information, call the help:
~# python3 pan20-celebrity-profiling-ngram-baseline.py --help
"""
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
from pathlib import Path
import logging
from pre_process import TRAIN_COUNT, TEST_COUNT, dataset_path
from sklearn.metrics import accuracy_score
import os
import numpy as np
import pickle


# Regular expressions for preprocessing
text_re = re.compile("[^a-zA-Z\s]")
url_re = re.compile("http(s)*://[\w]+\.(\w|/)*(\s|$)")
hashtag_re = re.compile("[\W]#[\w]*[\W]")
mention_re = re.compile("(^|[\W\s])@[\w]*[\W\s]")
smile_re = re.compile("(:\)|;\)|:-\)|;-\)|:\(|:-\(|:-o|:o|<3)")
emoji_re = re.compile("(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])")
not_ascii_re = re.compile("([^\x00-\x7F]+)")
time_re = re.compile("(^|\D)[\d]+:[\d]+")
numbers_re = re.compile("(^|\D)[\d]+[.'\d]*\D")
space_collapse_re = re.compile("[\s]+")

# numerical encoding of the classes
g_dict = {'male': 0, 'female': 1, 'nonbinary': 2}
inv_g_dict = {0: 'male', 1: 'female', 2: 'nonbinary'}
o_dict = {"sports": 0, "performer": 1, "creator": 2, "politics": 3, "science": 4, "manager": 5, "professional": 6, "religious": 7}
inv_o_dict = {0: "sports", 1: "performer", 2: "creator", 3: "politics", 4: "science", 5: "manager", 6: "professional", 7: "religious"}

# Hyperparameters
N_GRAM_RANGE = (1, 2)
MAX_WORD_FEATURES = 10000
MAX_TWEETS_PER_USER = 10000
MAX_FOLLOWERS_PER_CELEBRITY = 10


def _preprocess_feed(tweet: str):
    """ takes the original tweet text and returns the preprocessed texts.
    Preprocessing includes:
        - lowercasing
        - replacing hyperlinks with <url>, mentions with <user>, time, numbers, emoticons, emojis
        - removing additional non-ascii special characters
        - collapsing spaces
    """
    t = tweet.lower()
    t = re.sub(url_re, " <URL> ", t)
    t = t.replace("\n", "")
    t = t.replace("#", " <HASHTAG> ")
    t = re.sub(mention_re, " <USER> ", t)
    t = re.sub(smile_re, " <EMOTICON> ", t)
    t = re.sub(emoji_re, " <EMOJI> ", t)
    t = re.sub(time_re, " <TIME> ", t)
    t = re.sub(numbers_re, " <NUMBER> ", t)
    t = re.sub(not_ascii_re, "", t)
    t = re.sub(space_collapse_re, " ", t)
    t = t.strip()
    return t


def _read_text_linewise(p, mode):
    """ load each celebrity, concat the first 500 tweets and add a separator token between each """
    if mode == 'celeb':
        for line in open(p, "r"):
            yield " <eotweet> ".join(json.loads(line)["text"][:MAX_TWEETS_PER_USER])
    elif mode == 'follow':
        count = 0
        for line in open(p, "r", encoding="utf8"):
            if count == TRAIN_COUNT + TEST_COUNT:
                break
            count += 1
            yield " <eofollower> ".join([" <eotweet> ".join(follower[:MAX_TWEETS_PER_USER])
                                         for follower in json.loads(line)["text"][:MAX_FOLLOWERS_PER_CELEBRITY]])


def _get_age_class(by):
    """ convert the birthyears of a certain range to the center point.
     This is to reduce the number of classes when doing a classification model over regression on age
     """
    by = int(by)
    if 1940 <= by <= 1955:
        return 1947
    elif 1956 <= by <= 1969:
        return 1963
    elif 1970 <= by <= 1980:
        return 1975
    elif 1981 <= by <= 1989:
        return 1985
    elif 1990 <= by <= 1999:
        return 1994


def load_dataset(dataset_path: str, mode: str, vectorizer_path: str):
    """
    load the dataset, preprocess the texts for ML and build a feature matrix
    :param dataset_path: path to the dataset to be loaded
    :param mode: 'celeb' or 'follow' to load the celebrity feed or the follower feeds
    :param vectorizer_path: Path to a stored vectorizer which will be loaded from there if available or created and
                            stored there.
    :return: x, - x is the feature matrix for the texts
             y_age, y_gender, y_occ, - y are the targets for each labels
             ids - the ids identifying the indices of x and y
    """
    if mode == "celeb":
        x_path = dataset_path + "/celebrity-feeds.ndjson"
    else:
        x_path = dataset_path + "/feeds.ndjson"
    y_data = [json.loads(line) for line in open("./data/gt-labels.ndjson", "r")]

    if not Path(vectorizer_path).exists():
        logging.info("no stored vectorizer found, creating ...")
        vec = TfidfVectorizer(preprocessor=_preprocess_feed, ngram_range=N_GRAM_RANGE,
                              max_features=MAX_WORD_FEATURES, analyzer='word', min_df=3,
                              )# norm='l1')
        vec.fit(_read_text_linewise(x_path, mode))
        joblib.dump(vec, vectorizer_path)
    else:
        logging.info("loading stored vectorizer")
        vec = joblib.load(vectorizer_path)

    # load x data
    logging.info("transforming data ...")
    x = vec.transform(_read_text_linewise(x_path, mode))

    # load Y data
    # y_gender = [g_dict[l["gender"]] for l in y_data]
    y_gender = []
    y_occ = []
    y_age = []
    ids = []

    for l in y_data:
        y_gender.append(g_dict[l["gender"]])
        y_occ.append(o_dict[l["occupation"]])
        y_age.append(_get_age_class(l["birthyear"]))
        ids.append(l["id"])

    # y_occ = [o_dict[l["occupation"]] for l in y_data]
    # y_age = [_get_age_class(l["birthyear"]) for l in y_data]
    # ids = [i["id"] for i in y_data]
    return x, y_age, y_gender, y_occ, ids


def logreg(mode, vectorizer, training_dir):
    """ Main method for the baselines. It wires data loading, model training, and evaluation.
    The model used is a simple, linear LogisticRegression from sklearn.
    The method predicts on the given test dataset and writes the results to a labels.ndjson.
    Use the evaluator from the celebrity profiling task at PAN2020 to evaluate the results.
    """
    # 1. load the training dataset
    NORMALIZE = True
    pre_load = True

    logging.basicConfig(level=logging.INFO)
    logging.info("loading training dataset")
    if not pre_load:
        x, y_age, y_gender, y_occ, cid = \
            load_dataset(training_dir, mode, vectorizer)

        x_train = x[0:TRAIN_COUNT, :]

        y_train_age = y_age[0:TRAIN_COUNT]
        y_train_gender = y_gender[0:TRAIN_COUNT]
        y_train_occ = y_occ[0:TRAIN_COUNT]

        x_test = x[TRAIN_COUNT:TRAIN_COUNT+TEST_COUNT, :]
        y_test_age = y_age[TRAIN_COUNT:]
        y_test_gender = y_gender[TRAIN_COUNT:]
        y_test_occ = y_occ[TRAIN_COUNT:]
        cid = cid[TRAIN_COUNT:]

        if NORMALIZE:
            x_train = normalize(x_train, axis=1, norm='l1')
            x_test = normalize(x_test, axis=1, norm='l1')

        data_path = 'data/loaded_data.npz'
        with open(data_path, 'wb') as f:
            pickle.dump([x_train, y_train_age, y_train_gender, y_train_occ, x_test, y_test_age, y_test_gender, y_test_occ, cid], f)

    else:
        data_path = 'data/loaded_data.npz'
        if os.path.isfile(data_path):
            with open(data_path, 'rb') as f:
                x_train, y_train_age, y_train_gender, y_train_occ, x_test, y_test_age, y_test_gender, y_test_occ, cid = pickle.load(f)
    # exit()
    # 2. train models
    y_train_age = [x if isinstance(x, int) else 0 for x in y_train_age]
    y_test_age = [x if isinstance(x, int) else 0 for x in y_test_age]
    logging.info("fitting model age")
    # age_model = LogisticRegression(multi_class='multinomial', solver="newton-cg")
    # age_model = SVC()
    # age_model = DecisionTreeClassifier()
    age_model = RandomForestClassifier(n_estimators=15)
    # age_model = MultinomialNB()
    age_model.fit(x_train, y_train_age)
    logging.info("fitting model gender")
    # gender_model = LogisticRegression(multi_class='multinomial', solver="newton-cg")
    # gender_model = SVC(verbose=True, C=10, class_weight={0: 10, 1:1})
    # gender_model = DecisionTreeClassifier()
    gender_model = RandomForestClassifier(n_estimators=15)
    # gender_model = MultinomialNB()
    gender_model.fit(x_train, y_train_gender)
    logging.info("fitting model acc")
    # occ_model = LogisticRegression(multi_class='multinomial', solver="newton-cg")
    # occ_model = SVC(verbose=True)
    # occ_model = DecisionTreeClassifier()
    occ_model = RandomForestClassifier(n_estimators=15)
    # occ_model = MultinomialNB()
    occ_model.fit(x_train, y_train_occ)

    # 3. load the test dataset
    logging.info("loading test dataset ...")
    # x_test, y_test_age, y_test_gender, y_test_occ, cid = \
    #     load_dataset(test_dir, mode, vectorizer)

    # 4. Predict and Evaluate
    logging.info("predicting")
    age_pred = age_model.predict(x_test)
    gender_pred = gender_model.predict(x_test)
    occ_pred = occ_model.predict(x_test)

    # gender_pred = gender_model.predict(x_train)
    # occ_pred = occ_model.predict(x_train)
    output_labels = [{"id": i, "occupation": inv_o_dict[o], "gender": inv_g_dict[g], "birthyear": int(a) }
                     for i, o, g, a in zip(cid, occ_pred, gender_pred, age_pred)]
    # output_labels = [{"id": i, "gender": inv_g_dict[g], "occupation": inv_o_dict[o]}
    #                  for i, g, o in zip(cid, gender_pred, occ_pred)]

    if not os.path.isdir('./results'):
        os.makedirs('./results')

    open("./results/all-predictions.ndjson", "w").writelines(
        [json.dumps(x) + "\n" for x in output_labels]
    )

    pred_dict = {"prediction": output_labels[0:10]}
    with open('./results/pred.json', 'w') as outfile:
        json.dump(pred_dict, outfile)

    gt_labels = [{"id": i, "occupation": inv_o_dict[o], "gender": inv_g_dict[g], "birthyear": int(a) }
                 for i, o, g, a in zip(cid, y_test_occ, y_test_gender, y_test_age)]
    gt_dict = {"ground_truth": gt_labels[0:10]}
    with open('./results/gt.json', 'w') as outfile:
        json.dump(gt_dict, outfile)

    # saving trained models
    if not os.path.isdir("./pretrained-models"):
        os.makedirs("./pretrained-models")

    pickle.dump(age_model, open("./pretrained-models/age-model", 'wb'))
    pickle.dump(gender_model, open("./pretrained-models/gender-model", 'wb'))
    pickle.dump(occ_model, open("./pretrained-models/occ-model", 'wb'))

    print("Accuracy for age model: {:.2f}%".format(accuracy_score(age_pred, y_test_age) * 100.0))

    print("Accuracy for gender model: {:.2f}%".format(accuracy_score(gender_pred, y_test_gender) * 100.0))

    print("Accuracy for occupation model: {:.2f}%".format(accuracy_score(occ_pred, y_test_occ) * 100.0))


if __name__ == "__main__":
    mode = "follow"
    vectorizer = "./data/celeb-word-vectorizer.joblib"
    logreg(mode, vectorizer, dataset_path)
