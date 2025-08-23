### Created by Myongin Oh
### Last updated on Aug 23, 2025

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization
import multiprocessing
import time

cores=multiprocessing.cpu_count()
print("cores:", cores)

n_estimators=10

start_time = time.time()

class SVMFeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.99):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha

    def _evaluate(self, x):
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        clf = SVC(kernel='linear', C=1, decision_function_shape='ovr')
        accuracy = cross_val_score(clf, self.X_train[:, selected], self.y_train, cv=5, n_jobs=-1).mean()
        score = 1 - accuracy
        num_features = self.X_train.shape[1]
        return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)

df = pd.read_csv('BPSO_01_iter.csv') # Use the output file obtained after feature selection
feature_names = df.columns.values

zeroList = [0]*10000 # class 1
oneList = [1]*10000 # class 2
twoList = [2]*10000 # class 3

X = df.values
y = np.array(zeroList + oneList + twoList)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
problem = SVMFeatureSelection(X_train, y_train)
task = Task(problem, max_iters=500)
algorithm = ParticleSwarmOptimization(population_size=100, seed=1234)
best_features, best_fitness = algorithm.run(task)

selected_features = best_features > 0.5
print('Number of selected features:', selected_features.sum())
print('Selected features:', ', '.join(feature_names[selected_features]))

print("--- %s seconds ---" % (time.time() - start_time))

model_selected_simple = SVC(kernel='linear', C=1, decision_function_shape='ovr')
model_all_simple = SVC(kernel='linear', C=1, decision_function_shape='ovr')
model_selected_bagging = OneVsRestClassifier(BaggingClassifier(LinearSVC(dual=False), n_jobs=1, n_estimators=10, max_samples=1.0/n_estimators), n_jobs=1)
model_all_bagging = OneVsRestClassifier(BaggingClassifier(LinearSVC(dual=False), n_jobs=1, n_estimators=10, max_samples=1.0/n_estimators), n_jobs=1)

model_selected_simple.fit(X_train[:, selected_features], y_train)
print('Subset accuracy on test (simple svm):', model_selected_simple.score(X_test[:, selected_features], y_test))
accuracy_cv_simple = cross_val_score(model_selected_simple, X_test[:, selected_features], y_test, cv=5, n_jobs=1).mean()
print('Subset accuracy on test cv (simple svm):', accuracy_cv_simple)
model_all_simple.fit(X_train, y_train)
print('All Features Accuracy on test (simple svm):', model_all_simple.score(X_test, y_test))

model_selected_bagging.fit(X_train[:, selected_features], y_train)
print('Subset accuracy on test (bagging svm):', model_selected_bagging.score(X_test[:, selected_features], y_test))
accuracy_cv_bagging = cross_val_score(model_selected_bagging, X_test[:, selected_features], y_test, cv=5, n_jobs=1).mean()
print('Subset accuracy on test cv (bagging svm):', accuracy_cv_bagging)
model_all_bagging.fit(X_train, y_train)
print('All Features Accuracy on test (bagging svm):', model_all_bagging.score(X_test, y_test))

print(feature_names[selected_features])
dfRED = df[feature_names[selected_features]]
dfRED.to_csv('BPSO_02_iter.csv', encoding='utf-8', index=False)
