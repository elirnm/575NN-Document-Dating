import pickle
import sys
from pprint import pprint
from time import perf_counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# load data
with open("../cached_data/test_docs.pkl", 'rb') as f:
    test_docs = pickle.load(f)
with open("../cached_data/devtest_docs.pkl", 'rb') as f:
    devtest_docs = pickle.load(f)
with open("../cached_data/train_docs.pkl", 'rb') as f:
    train_docs = pickle.load(f)

def get_label(doc):
    lab = doc.corpus_section[:2]
    if lab[1] == "X" and len(doc.corpus_section) == 4:
        lab = lab[0] + doc.corpus_section[3]
    return lab.strip()

train_data = [x.cleaned_text for x in train_docs]
train_labels = [get_label(x) for x in train_docs]

devtest_data = [x.cleaned_text for x in devtest_docs]
devtest_labels = [get_label(x) for x in devtest_docs]

test_data = [x.cleaned_text for x in test_docs]
test_labels = [get_label(x) for x in test_docs]

# convert labels to numbers
label_vals = {}
val_labels = {}
for i, label in enumerate(set(train_labels + devtest_labels + test_labels)):
    val_labels[i] = label
    label_vals[label] = i

label_names = sorted(label_vals.keys(), key=lambda x: label_vals[x])

train_labels = [label_vals[x] for x in train_labels]
devtest_labels = [label_vals[x] for x in devtest_labels]
test_labels = [label_vals[x] for x in test_labels]

# convert data to count vectors
vec = CountVectorizer(analyzer='char')
start_time = perf_counter()
train_count = vec.fit_transform(train_data)
train_count_time = perf_counter()

devtest_count = vec.transform(devtest_data)
test_count = vec.transform(test_data)
# test_count_time = perf_counter()

# create all of our classifiers
mnb = MultinomialNB()
maxent = LogisticRegression()
svm = SVC()
dtree = DecisionTreeClassifier()

# train all our classifiers
train_start_time = perf_counter()
mnb.fit(train_count, train_labels)
mnb_train_time = perf_counter()

maxent.fit(train_count, train_labels)
maxent_train_time = perf_counter()

svm.fit(train_count, train_labels)
svm_train_time = perf_counter()

dtree.fit(train_count, train_labels)
dtree_train_time = perf_counter()

# devtest all our classifers
mnb_devres = mnb.predict(devtest_count)
maxent_devres = maxent.predict(devtest_count)
svm_devres = svm.predict(devtest_count)
dtree_devres = dtree.predict(devtest_count)

# test all our classifers
mnb_res = mnb.predict(test_count)
maxent_res = maxent.predict(test_count)
svm_res = svm.predict(test_count)
dtree_res = dtree.predict(test_count)

# evaluate performance
# results = {}
# results["Multinomial Naive Bayes"] = classification_report(test_labels, mnb_res, target_names=label_names)
# results["MaxEnt"] = classification_report(test_labels, maxent_res, target_names=label_names)
# results["SVM"] = classification_report(test_labels, svm_res, target_names=label_names)
# results["Decision Tree"] = classification_report(test_labels, dtree_res, target_names=label_names)

devaccs = {}
devaccs["Multinomial Naive Bayes"] = accuracy_score(devtest_labels, mnb_devres)
devaccs["MaxEnt"] = accuracy_score(devtest_labels, maxent_devres)
devaccs["SVM"] = accuracy_score(devtest_labels, svm_devres)
devaccs["Decision Tree"] = accuracy_score(devtest_labels, dtree_devres)

accs = {}
accs["Multinomial Naive Bayes"] = accuracy_score(test_labels, mnb_res)
accs["MaxEnt"] = accuracy_score(test_labels, maxent_res)
accs["SVM"] = accuracy_score(test_labels, svm_res)
accs["Decision Tree"] = accuracy_score(test_labels, dtree_res)

# conf_matrix = {}
# conf_matrix["Multinomial Naive Bayes"] = confusion_matrix(test_labels, mnb_res)
# conf_matrix["MaxEnt"] = confusion_matrix(test_labels, maxent_res)
# conf_matrix["SVM"] = confusion_matrix(test_labels, svm_res)
# conf_matrix["Decision Tree"] = confusion_matrix(test_labels, dtree_res)

# assemble runtime data
train_count_time = (train_count_time - start_time)
train_times = {}
train_times["Multinomial Naive Bayes"] = (mnb_train_time - train_start_time) + train_count_time
train_times["MaxEnt"] = (maxent_train_time - mnb_train_time) + train_count_time
train_times["SVM"] = (svm_train_time - maxent_train_time) + train_count_time
train_times["Decision Tree"] = (dtree_train_time - svm_train_time) + train_count_time

for clf in accs:
    print(clf)
    print("Train time: {0} seconds".format(train_times[clf]))
    # print(results[clf])
    # print()
    print("Devtest accuracy: " + str(devaccs[clf]))
    # print()
    print("Test accuracy: " + str(accs[clf]))
    # print()
    # pprint(label_vals)
    # print()
    # print(conf_matrix[clf])
    # print()
    print("----------")
    print()
