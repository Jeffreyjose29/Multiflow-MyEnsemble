# -*- coding: utf-8 -*-
"""
Created on Wed Jun 3 21:22:13 2020

@author: jeffr

Note: A large majority of this code has been modified from the adaptive random forest implementation for scikit-multiflow.

"""
from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.metrics import ClassificationPerformanceEvaluator
from skmultiflow.metrics import WindowClassificationMeasurements
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import FileStream
from skmultiflow.data import WaveformGenerator
from sklearn.preprocessing import normalize

#utilisation imports
from skmultiflow.utils import get_dimensions, check_weights, normalize_values_in_dict
from skmultiflow.utils.utils import *
from skmultiflow.trees import HoeffdingTreeClassifier
from copy import deepcopy
#meta imports
from skmultiflow.meta import LeveragingBaggingClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from skmultiflow.meta import AccuracyWeightedEnsemble

#basic imports
import numpy as np
import warnings
import random

warnings.filterwarnings("ignore")

'''
stream = WaveformGenerator()
stream.prepare_for_use()

ht = HoeffdingTreeClassifier()

evaluator = EvaluatePrequential(show_plot=True,
                                pretrain_size=200,
                                max_samples=20000)

evaluator.evaluate(stream=stream, model=ht)

'''

'''
The three parameters taken by my ensembles is:
    s = The number of learners
    l = The length of the window
    random_seed = Used to initialise a random object
'''
def MyEnsemble(s = 3, l = 100, random_seed = 1):
    return MyEnsembleClassifier(s = s, l = l, random_seed = random_seed)

class MyEnsembleClassifier(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    def __init__(self, s = 3, l = 100, random_seed = 1):
        super().__init__()
        #self.base_estimator = base_estimator
        self.s = s
        self.l = l
        #self.split_confidence = split_confidence
        self.random_seed = random_seed
        random.seed(self.random_seed)
        self.ensemble = None
        self.instances_seen = 0
        
        self.classes = None
        
    def partial_fit(self, X, y, classes = None, sample_weight = None):
        """ Partially (incrementally) fit the model.
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.
        y: numpy.ndarray of shape (n_samples)
            An array-like with the class labels of all samples in X.
        classes: numpy.ndarray, list, optional (default=None)
            Array with all possible/known class labels. This is an optional parameter, except
            for the first partial_fit call where it is compulsory.
        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed.
        """
        if self.classes is None and classes is not None:
            self.classes = classes
        
        if sample_weight is None:
            weight = 1.0 
        else: 
            weight = sample_weight
            
        
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            weight = check_weights(weight, expand_length = row_cnt)
            for iterator in range(row_cnt):
                if weight[iterator] != 0.0:
                    self._partial_fit(X[iterator], y[iterator], self.classes, weight[iterator])                    
        return self
        
    def predict_proba(self, X):
        """ Estimates the probability of each sample in X belonging to each of the class-labels.
        Class probabilities are calculated as the mean predicted class probabilities
        per base estimator.
        Parameters
        ----------
         X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the class probabilities.
        Returns
        -------
        numpy.ndarray of shape (n_samples, n_classes)
            Predicted class probabilities for all instances in X.
            If class labels were specified in a `partial_fit` call, the order of the columns
            matches `self.classes`.
            If classes were not specified, they are assumed to be 0-indexed.
            Class probabilities for a sample shall sum to 1 as long as at least one estimators
            has non-zero predictions.
            If no estimator can predict probabilities, probabilities of 0 are returned.
        """
        if self.ensemble is None:
            self.ensemble = [self._init_ensemble_member() for _ in range(self.s)]

        y_proba_sum = None
        for i in range(self.s):
            if self.ensemble[i].prediction != 0:
                accuracy = self.ensemble[i].correct / self.ensemble[i].prediction
            else:
                accuracy = 0
            y_proba_current = self.ensemble[i].predict_proba(X) * accuracy
            if y_proba_sum is None:
                y_proba_sum = y_proba_current
            else:
                y_proba_sum = y_proba_sum + y_proba_current

        return normalize(y_proba_sum, norm='l1') 
        
            
    def reset(self):
        print("Not implemented")
            
    def predict(self, X):
        """ Predict classes for the passed data.
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the class labels for.
        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.
        """
        y_proba = self.predict_proba(X)
        n_rows = y_proba.shape[0]
        y_pred = np.zeros(n_rows, dtype=int)
        for i in range(n_rows):
            index = np.argmax(y_proba[i])
            y_pred[i] = index
        return y_pred
        
        
    def _partial_fit(self, X, y, classes=None, sample_weight=1.0):
        low = 0
        lowest_prediction = 0
        self.instances_seen += 1
        
        if self.ensemble is None:
            self.ensemble = [self._init_ensemble_member() for _ in range(self.s)]
        
        index = self.instances_seen % self.l
                    
        if index == 1:
            self.ensemble.append(self._init_ensemble_member())
            
        for iterator in range(self.s + 1):
            self.ensemble[iterator].prediction += sample_weight
            y_pred = self.ensemble[iterator].predict(np.asarray([X]))
            
            if abs(y_pred - y) < 0.000000002:
                self.ensemble[iterator].correct += sample_weight
                
            self.ensemble[iterator].partial_fit(np.asarray([X]), np.asarray([y]), classes = classes, sample_weight = np.asarray([sample_weight]))
            
        if index == 0:
            for iterator, classifier in enumerate(self.ensemble):
            
                if classifier.prediction != 0:
                    accuracy = classifier.correct / classifier.prediction
                else:
                    accuracy = 0
                    
                lowest_prediction = self.ensemble[low]
                lowest_accuracy = lowest_prediction.correct / lowest_prediction.prediction
                if accuracy <= lowest_accuracy:
                    low = iterator
            
            self.ensemble.pop(low)
            
        
    def get_votes_for_instance(self, X):
        if self.ensemble is None:
            self.ensemble = [self._init_ensemble_member() for _ in range(self.s)]
        combined_votes = {}

        for i in range(self.s):
            vote = deepcopy(self.ensemble[i].instance_votes(X))
            if vote != {} and sum(vote.values()) > 0:
                vote = normalize_values_in_dict(vote, inplace=False)
                if self.ensemble.prediction != 0:
                    accuracy = self.ensemble[i].correct / self.ensemble[i].prediction
                else:
                    accuracy = 0
                if accuracy != 0.0:
                    for k in vote:
                        vote[k] = vote[k] * accuracy

                # Add values
                for k in vote:
                    try:
                        combined_votes[k] += vote[k]
                    except KeyError:
                        combined_votes[k] = vote[k]
        return combined_votes
    
    def _init_ensemble_member(self):
        #randomise the hoeffding tree three parameters
        grace_period = (random.randint(0, 20) + 1) * 10  
        split_confidence = (random.randint(0, 20) + 1) * 0.05  
        tie_threshold = (random.randint(0, 20) + 1) * 0.05  
        return MyEnsembleBaseLearner(classifier=HoeffdingTreeClassifier(grace_period=grace_period, split_confidence=split_confidence, tie_threshold=tie_threshold))        

class MyEnsembleBaseLearner(BaseSKMObject):
    def __init__(self, classifier: HoeffdingTreeClassifier):
        self.classifier = classifier
        self.prediction = 0
        self.correct = 0
    
    def partial_fit(self, X, y, classes, sample_weight):
        self.classifier.partial_fit(X, y, classes=classes, sample_weight=sample_weight)

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_votes_for_instance(self, X):
        return self.classifier.get_votes_for_instance(X)

    def reset(self):
        self.classifier.reset()
        
        
electricity = "C:/Users/jeffr/OneDrive/Desktop/DataStreamAssignmentTwo/dataset/electricity.csv"
covertype = "C:/Users/jeffr/OneDrive/Desktop/DataStreamAssignmentTwo/dataset/covtype.csv"
rtg_abrupt = "C:/Users/jeffr/OneDrive/Desktop/DataStreamAssignmentTwo/dataset/RTG_2abrupt.csv"
sea_gradual = "C:/Users/jeffr/OneDrive/Desktop/DataStreamAssignmentTwo/dataset/SEA_gradual.csv"
sea_abrupt = "C:/Users/jeffr/OneDrive/Desktop/DataStreamAssignmentTwo/dataset/SEA_abrupt.csv"
real_data = "C:/Users/jeffr/OneDrive/Desktop/STATS_FINAL_PROJECT/INLA Project/lowBirthWeight.csv"

# Experiment 1: Changing the number of learners while keep the value of l constant at 1000
'''
ensemble_stream = FileStream(covertype)
ensemble_stream.prepare_for_use()
ensemble_stream.next_sample(10)
ensemble_stream.n_remaining_samples()

metrics = ['accuracy', 'running_time']

ht = HoeffdingTreeClassifier()
me_s5 = MyEnsemble(s = 5, l = 1000,random_seed = 1000)  
me_s10 = MyEnsemble(s = 10, l = 1000, random_seed = 1000)
me_s20 = MyEnsemble(s = 20, l = 1000, random_seed = 1000)
me_s30 = MyEnsemble(s = 30, l = 1000, random_seed = 1000)


evaluator = EvaluatePrequential(show_plot=False,
                                n_wait = 100, max_samples = 50000,
                                metrics = metrics)

evaluator.evaluate(stream=ensemble_stream, model=[ht, me_s5, me_s10, me_s20, me_s30], 
                   model_names = ['HT','ME_S5', 'ME_S10', 'ME_S20', 'ME_S30'])
'''
# Experiment 2: Keeping s = 20 and l = 1000 but changing up the random seed{1, 2, 3, 4}
'''
ensemble_stream = FileStream(sea_gradual)
ensemble_stream.prepare_for_use()
ensemble_stream.next_sample(10)
ensemble_stream.n_remaining_samples()

metrics = ['accuracy', 'running_time']

me_s1 = MyEnsemble(s = 20, l = 1000,random_seed = 1)  
me_s2 = MyEnsemble(s = 20, l = 1000, random_seed = 2)
me_s3 = MyEnsemble(s = 20, l = 1000, random_seed = 3)
me_s4 = MyEnsemble(s = 20, l = 1000, random_seed = 4)
me_s5 = MyEnsemble(s = 20, l = 1000, random_seed = 5)



evaluator = EvaluatePrequential(show_plot=False,
                                n_wait = 100, max_samples = 75000,
                                metrics = metrics)

evaluator.evaluate(stream=ensemble_stream, model=[me_s1, me_s2, me_s3, me_s4, me_s5], 
                   model_names = ['ME_S1', 'ME_S2', 'ME_S3', 'ME_S4', 'ME_S5'])
'''
# Experiment 3: Keeping s = 20 and varying the levels of the window
'''
ensemble_stream = FileStream(sea_abrupt)
ensemble_stream.prepare_for_use()
ensemble_stream.next_sample(10)
ensemble_stream.n_remaining_samples()

metrics = ['accuracy', 'running_time']

me_l500 = MyEnsemble(s = 20, l = 500,random_seed = 1000)  
me_l1000 = MyEnsemble(s = 20, l = 1000, random_seed = 1000)
me_l2000 = MyEnsemble(s = 20, l = 2000, random_seed = 1000)
me_l5000 = MyEnsemble(s = 20, l = 5000, random_seed = 1000)
me_l10000 = MyEnsemble(s = 20, l = 10000, random_seed = 1000)


evaluator = EvaluatePrequential(show_plot=False,
                                n_wait = 100, max_samples = 75000,
                                metrics = metrics)

evaluator.evaluate(stream=ensemble_stream, model=[me_l500, me_l1000, me_l2000, me_l5000, me_l10000], 
                   model_names = ['ME_L500', 'ME_L1000', 'ME_L2000', 'ME_L5000', 'ME_L10000'])
'''

# Experiment 2: MyEnsembleClassifier vs. Others
'''
ensemble_stream = FileStream(covertype)
ensemble_stream.prepare_for_use()
ensemble_stream.next_sample(10)
ensemble_stream.n_remaining_samples()

metrics = ['accuracy', 'running_time']

me = MyEnsemble(s = 20, l = 100,random_seed = 1000)  
lbc = LeveragingBaggingClassifier(n_estimators = 20)
dmc = DynamicWeightedMajorityClassifier(n_estimators = 20)
arf = AdaptiveRandomForestClassifier(n_estimators = 20, lambda_value = 0.6)


evaluator = EvaluatePrequential(show_plot=False,
                                n_wait = 100, max_samples = 30000,
                                metrics = metrics)

evaluator.evaluate(stream=ensemble_stream, model=[me, lbc, dmc, arf], 
                   model_names = ['ME:', 'LBC:', 'DMC:', 'ARF:'])
'''

# Looking at the concept drift detection for experiment one part c:
'''
ensemble_stream = FileStream(sea_gradual)
ensemble_stream.prepare_for_use()
ensemble_stream.next_sample(10)
ensemble_stream.n_remaining_samples()

metrics = ['accuracy', 'running_time']

me = MyEnsemble(s = 30, l = 1000,random_seed = 1000)  

evaluator = EvaluatePrequential(show_plot=True,
                                n_wait = 100, max_samples = 75000,
                                metrics = metrics)

evaluator.evaluate(stream=ensemble_stream, model=[me], 
                   model_names = ['ME:'])
'''
