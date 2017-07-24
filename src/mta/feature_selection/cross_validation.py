import numpy
from copy import copy
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
            self._model_folds = []
            for fold_id in range(self.n_fold):
                merged_dataset = self._merge_exclude(fold_id)
                train_model = copy(self.model)
                train_model.load_dataset(merged_dataset)
                train_model.fit()
                self._model_folds.append(train_model)
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
        scores = numpy.zeros(self.n_fold)
        for fold_id in range(self.n_fold):
            rating_list =  self._dataset_folds[fold_id].ratings.to_list() 
            predictions = self._model_folds[fold_id].predict()
            scores[fold_id] = metirc_func(rating_list,predictions)
        if self.verbose:
            print('scores: ',scores) 
        return scores

    def select(self,metirc_func):
        scores = self.score(metirc_func)
        min_score = min(scores)
        best_fold_id = 0
        for fold_id in range(self.n_fold):
            if scores[fold_id] == min_score:
                best_fold_id = fold_id
                break
        return copy(self._model_folds[best_fold_id])










