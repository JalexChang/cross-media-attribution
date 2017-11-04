import numpy
from copy import copy
import time
from mta.model import *
from mta.dataset import Dataset


class CrossValidation:
    def __init__(self, verbose = False):
        self.model = None
        self._model_folds = None
        self.dataset = None
        self._datset_folds = None
        self.n_fold = 0
        self.verbose = verbose

    def load(self, dataset, model):
        self.dataset = dataset
        self.model = model
        if self.verbose:
            print('load dataset: ', self.dataset.matrix_shape())
            print('load learning model: ',type(self.model))
        return self

    def split(self,n_fold=5):
        self.n_fold = n_fold
        self._dataset_folds = Dataset.kfolds(self.dataset,n_fold)
        if self.verbose:
            print('split dataset into ',n_fold,' folds')
        return self


    def run(self):
        if self._dataset_folds is not None:
            if self._model_folds is None:
                self._model_folds = []
                for fold_id in range(self.n_fold):
                    merged_dataset = self._merge_exclude(fold_id)
                    train_model = copy(self.model)
                    train_model.load_dataset(merged_dataset)
                    train_model.fit()
                    self._model_folds.append(train_model)
                    if self.verbose:
                        print ('fold ',fold_id+1,' is trained')
            else:
                for fold_id in range(self.n_fold):
                    self._model_folds[fold_id].fit()
                    if self.verbose:
                        print ('fold ',fold_id+1,' is trained')

    def _merge_exclude(self,exclude_fold_id):
        merged_dataset_l = Dataset.merge(self._dataset_folds[:exclude_fold_id]) if len(self._dataset_folds[:exclude_fold_id])>0 else None
        merged_dataset_r = Dataset.merge(self._dataset_folds[exclude_fold_id+1:]) if len(self._dataset_folds[exclude_fold_id+1:])>0 else None

        merged_dataset = None;
        if merged_dataset_l is None:
            merged_dataset =  merged_dataset_r
        elif merged_dataset_r is None:
            merged_dataset =  merged_dataset_l;
        else:
            merged_dataset = Dataset.append(merged_dataset_l, merged_dataset_r)
        return merged_dataset
 
    def score(self,metirc_func):
        return self._score(metirc_func, "normal")

    def score_avg(self,metirc_func):
        return self._score(metirc_func, "avg")

    def _score(self, metirc_func, score_type="normal"):
        scores = numpy.zeros(self.n_fold)
        for fold_id in range(self.n_fold):
            begin_time = time.time()
            rating_list =  self._dataset_folds[fold_id].ratings.to_list()
            predictions = self._get_predictions(fold_id, rating_list, score_type)
            scores[fold_id] = metirc_func(rating_list,predictions)
            end_time = time.time()
            if self.verbose:
                print('scored fold',fold_id,'in', str(end_time - begin_time)) 
        return scores

    def _get_predictions(self, fold_id, rating_list, score_type="normal"):
        if score_type == "normal":
            return self._model_folds[fold_id].predict(rating_list)
        elif score_type =="avg":
            return self._model_folds[fold_id].predict_average(rating_list)

    def predict(self):
        matrix_pred = numpy.zeros(self.dataset.ratings.matrix_shape())
        for index in range(len(self._model_folds)):
            rating_list = self._dataset_folds[index].ratings.to_list()
            pred_ratings = self._get_predictions(index, rating_list, "normal")
            for u_id, i_id, rating in rating_list:
                matrix_pred[u_id][i_id] += pred_ratings[u_id][i_id]
        return matrix_pred 
        
    def select(self,metirc_func):
        scores = self.score(metirc_func)
        min_score = min(scores)
        best_fold_id = 0
        for fold_id in range(self.n_fold):
            if scores[fold_id] == min_score:
                best_fold_id = fold_id
                break
        return copy(self._model_folds[best_fold_id])










