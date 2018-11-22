# Submitted by
# Saraswathi Shanmugamoorthy
# CS 6375.003
# Assignment 2 - Spam detection using Logistic Regression

import collections
import re
import math
import copy
import os
import sys


# To save or store emails as dictionaries.
train_set = dict()
testing_set = dict()

# Filtered sets without stop words
train_set_filtered = dict()
testing_set_filtered = dict()

# stop words list
stop_words = []

train_set_vocabwords = []
train_set_filtered_vocabwords = []

# weights are stored as dictionary.
weight = {'weight_zero': 0.0}
weight_filtered = {'weight_zero': 0.0}


classes = ["ham", "spam"]

# learning const, total number of iteration, penalty that is lambda constant
learn_const = .01
penalty = 0.0
iteration_num = 50


def dataSetsMaking(store_dict, directory, classes_true):
    for directory_entry in os.listdir(directory):
        directory_entry_path = os.path.join(directory, directory_entry)
        if os.path.isfile(directory_entry_path):
            with open(directory_entry_path, 'r') as text_file:
                text = text_file.read()
                store_dict.update({directory_entry_path: Document(text, wordsFreq(text), classes_true)})



def getVocabWords(data_sets):
    v = []
    for i in data_sets:
        for j in data_sets[i].getWordFrequency():
            if j not in v:
                v.append(j)
    return v



def settingStopWords():
    stop = []
    with open('stop_words.txt', 'r') as txt:
        stop = (txt.read().splitlines())
    return stop


# To remove the stop words
def stopWordsRemoving(stop, data_sets):
    data_sets_filtered = copy.deepcopy(data_sets)
    for i in stop:
        for j in data_sets_filtered:
            if i in data_sets_filtered[j].getWordFrequency():
                del data_sets_filtered[j].getWordFrequency()[i]
    return data_sets_filtered


# For counting frequency of the words
def wordsFreq(text):
    wordsFrequency = collections.Counter(re.findall(r'\w+', text))
    return dict(wordsFrequency)


# For learning weights
def weightsLearning(train, parameters_weight, iteration, lam):
    for x in range(0, iteration):
        print x
        counter = 1
        for w in parameters_weight:
            sum = 0.0
            for i in train:
                sample_y = 0.0
                if train[i].getClassesTrue() == classes[1]:
                    sample_y = 1.0
                if w in train[i].getWordFrequency():
                    sum += float(train[i].getWordFrequency()[w]) * (sample_y - calcConditionalProb(classes[1], parameters_weight, train[i]))
            parameters_weight[w] += ((learn_const * sum) - (learn_const * float(lam) * parameters_weight[w]))


# Calculations for the conditional probability
def calcConditionalProb(class_probabilities, parameters_weight, doc):
   
    if class_probabilities == classes[0]:
        wx_0_sum = parameters_weight['weight_zero']
        for i in doc.getWordFrequency():
            if i not in parameters_weight:
                parameters_weight[i] = 0.0
            wx_0_sum += parameters_weight[i] * float(doc.getWordFrequency()[i])
        return 1.0 / (1.0 + math.exp(float(wx_0_sum)))


    elif class_probabilities == classes[1]:
        wx_1_sum = parameters_weight['weight_zero']
        for i in doc.getWordFrequency():
            if i not in parameters_weight:
                parameters_weight[i] = 0.0
            wx_1_sum += parameters_weight[i] * float(doc.getWordFrequency()[i])
        return math.exp(float(wx_1_sum)) / (1.0 + math.exp(float(wx_1_sum)))


# Applying the logistic regression algorithm
def applyingLogisticReg(data_instances, parameters_weight):
    scores = {}
    scores[0] = calcConditionalProb(classes[0], parameters_weight, data_instances)
    scores[1] = calcConditionalProb(classes[1], parameters_weight, data_instances)
    if scores[1] > scores[0]:
        return classes[1]
    else:
        return classes[0]


class Document:
    text = ""
    word_frequency = {'weight_zero': 1.0}

    classes_true = ""
    learned_classes = ""

    def __init__(self, text, counter, classes_true):
        self.text = text
        self.word_freqs = counter
        self.classes_true = classes_true

    def getTexts(self):
        return self.text

    def getWordFrequency(self):
        return self.word_freqs

    def getClassesTrue(self):
        return self.classes_true

    def getLearnedClasses(self):
        return self.learned_classes

    def setLearnedClasses(self, guess):
        self.learned_classes = guess


def main(training_dir_spam, training_dir_ham, testing_dir_spam, testing_dir_ham, lam_const):

    dataSetsMaking(train_set, training_dir_spam, classes[1])
    dataSetsMaking(train_set, training_dir_ham, classes[0])
    dataSetsMaking(testing_set, testing_dir_spam, classes[1])
    dataSetsMaking(testing_set, testing_dir_ham, classes[0])
    penalty = lam_const

    # list of Stop words
    stop_words = settingStopWords()

    # filtered data sets
    train_set_filtered = stopWordsRemoving(stop_words, train_set)
    testing_set_filtered = stopWordsRemoving(stop_words, testing_set)

    train_set_vocabwords = getVocabWords(train_set)
    train_set_filtered_vocabwords = getVocabWords(train_set_filtered)

    # Initializing weights of train set
    for i in train_set_vocabwords:
        weight[i] = 0.0
    for i in train_set_filtered_vocabwords:
        weight_filtered[i] = 0.0

    print "****************************************"
    print " "
    print "Please wait for unfiltered training set iterations"
    weightsLearning(train_set, weight, iteration_num, penalty)
    print "****************************************"
    print " "
    print "Please wait for filtered training set iterations"
    weightsLearning(train_set_filtered, weight_filtered, iteration_num, penalty)


    # Applying the logistic regression algorithm on the test set
    correct_predicts_count = 0.0
    for i in testing_set:
        testing_set[i].setLearnedClasses(applyingLogisticReg(testing_set[i], weight))
        if testing_set[i].getLearnedClasses() == testing_set[i].getClassesTrue():
            correct_predicts_count += 1.0

    # Applying the logistic regression algorithm on the filtered test set
    correct_predicts_count_filtered = 0.0
    for i in testing_set_filtered:
        testing_set_filtered[i].setLearnedClasses(applyingLogisticReg(testing_set_filtered[i], weight_filtered))
        if testing_set_filtered[i].getLearnedClasses() == testing_set_filtered[i].getClassesTrue():
            correct_predicts_count_filtered += 1.0
    print "****************************************"
    print " "
    print "Number of correct predictions without filtering the stop words:\t%d/%s" % (correct_predicts_count, len(testing_set))
    print "Accuracy without filtering the stop words:\t\t\t%.4f%%" % (100.0 * float(correct_predicts_count) / float(len(testing_set)))
    print "****************************************"
    print " "
    print "Number of correct predictions after filtering the stop words:\t\t%d/%s" % (correct_predicts_count_filtered, len(testing_set_filtered))
    print "Accuracy after filtering the stop words:\t\t\t%.4f%%" % (100.0 * float(correct_predicts_count_filtered) / float(len(testing_set_filtered)))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
