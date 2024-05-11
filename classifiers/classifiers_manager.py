import numpy as np
from sklearn import svm, tree, metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle
from enum import Enum


class Classifiers(Enum):
    svm = 0
    knn = 1
    random_forest = 2
    decision_tree = 3
    adaboost = 4


class ClassifiersManager:
    def __init__(self):
        self.classifiers_names = {Classifiers.svm: "SVM",
                                  Classifiers.knn: "KNN",
                                  Classifiers.random_forest: "Random_Forest",
                                  Classifiers.decision_tree: "Decision_Tree",
                                  Classifiers.adaboost: "Adaboost"}

        self.svm_classifier = svm.SVC(kernel='poly', degree=3, C=500, gamma=7)
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5)
        self.random_forest_classifier = RandomForestClassifier(
            n_estimators=350)
        self.decision_tree_classifier = tree.DecisionTreeClassifier()
        self.adaboost_classifier = AdaBoostClassifier()

        self.classifiers = {Classifiers.svm: self.svm_classifier,
                            Classifiers.knn: self.knn_classifier,
                            Classifiers.random_forest: self.random_forest_classifier,
                            Classifiers.decision_tree: self.decision_tree_classifier,
                            Classifiers.adaboost: self.adaboost_classifier}

        self.classifiers_metrics = {}

    def train(self, classifier_enum, training_features, training_labels):
        classifier = self.classifiers[classifier_enum]
        classifier.fit(training_features, training_labels)
        self.__save_model(classifier_enum, classifier)

    def test(self, classifier_enum, testing_features, true_labels, target_names):
        predicted_labels = self.classifiers[classifier_enum].predict(
            testing_features)
        classifier_metrics = metrics.classification_report(
            true_labels, predicted_labels, target_names=target_names)
        return classifier_metrics

    def predict(self, classifier_enum, testing_features):
        predicted_labels = self.classifiers[classifier_enum].predict(
            testing_features)
        return predicted_labels

    def getClassificationReport(self, predicted_labels, true_labels, target_names):
        classifier_metrics = metrics.classification_report(
            true_labels, predicted_labels, target_names=target_names)
        return classifier_metrics

    def train_all_classifiers(self, training_features, training_labels):
        for classifier_enum, _ in self.classifiers.items():
            self.train(classifier_enum, training_features, training_labels)

    def test_all_classifiers(self, testing_features, true_labels, target_names):

        for classifier_enum, _ in self.classifiers.items():
            metrics = self.test(classifier_enum, testing_features,
                                true_labels, target_names)
            classifier_name = self.classifiers_names[classifier_enum]
            self.classifiers_metrics[classifier_name] = metrics

        return self.classifiers_metrics

    def print_metrics(self):
        for classifier, metrics in self.classifiers_metrics.items():
            print(classifier)
            print(metrics)

    def __save_model(self, classifier_enum, model):
        with open(self.classifiers_names[classifier_enum]+".pkl", 'wb') as file:
            pickle.dump(model, file)

    def load_model(self, classifier_enum):
        with open(self.classifiers_names[classifier_enum]+".pkl", 'rb') as file:
            self.classifiers[classifier_enum] = pickle.load(file)

    def load_all_classifiers(self):
        for classifier_enum, _ in self.classifiers.items():
            self.load_model(classifier_enum)
