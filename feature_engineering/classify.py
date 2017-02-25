from csv import DictReader, DictWriter
import re
import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'


class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))
############################################################
    list_tv_chars=[]
    with open("tv_shows_and_chars.txt") as fh:
        text=fh.read()
        lines=text.splitlines()
        for line in lines:
            list_tv_chars.append((str(line)).lower())
    #print(list_tv_chars)

###########################################################
    count_inner_train=0
    count_outer_train=0
    train_new=[]
    for x in train:
        count_outer_train+=1
        print(count_outer_train)

        sentence=(x[kTEXT_FIELD]).lower()

        for i in list_tv_chars:
            count_inner_train+=1
            pattern=re.compile(str(i))
            match=pattern.match(sentence)
            if match:
                sentence=sentence+" "+i+" "+i+" "+i
                print("MATCHMATCHMATCH_TRAIN"+str(count_inner_train)+"   "+str(count_outer_train))
        train_new.append(sentence)

    print(" LENGTH TRAIN LENGTH TRAIN LENGTH TRAIN LENGTH TRAIN LENGTH TRAIN LENGTH TRAIN LENGTH TRAIN")
    print(len(train_new))

#############################################################
    count_inner_test=0
    count_outer_test=0
    test_new=[]
    for x in test:
        count_outer_test+=1
        print(count_outer_test)

        sentence=(x[kTEXT_FIELD]).lower()


        for i in list_tv_chars:
            count_inner_test+=1
            pattern=re.compile(str(i))
            match=pattern.match(sentence)
            if match:
                sentence=sentence+" "+i+" "+i+" "+i
                print("MATCHMATCHMATCH_TEST"+str(count_inner_test)+"   "+str(count_outer_test))
        test_new.append(sentence)

    print(" LENGTH TEST LENGTH TEST LENGTH TEST LENGTH TEST LENGTH TEST LENGTH TEST LENGTH TEST")
    print(len(test_new))

##############################################################

    feat = Featurizer()
    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    print("Label set: %s" % str(labels))
    x_train = feat.train_feature(x for x in train_new)
    x_test = feat.test_feature(x for x in test_new)

    y_train = array(list(labels.index(x[kTARGET_FIELD]) for x in train))

    print(len(train), len(y_train))
    print(set(y_train))


    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    feat.show_top10(lr, labels)

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["Id", "spoiler"])
    o.writeheader()
    for ii, pp in zip([x['Id'] for x in test], predictions):
        d = {'Id': ii, 'spoiler': labels[pp]}
        o.writerow(d)
