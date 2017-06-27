import numpy
from copy import copy
from mta.model import *
from mta.dataset import Dataset


class CrossValidation:
    def __init__(self, verbose = False):
        self.model = None
        self.model_folds = None
        self.dataset = None
        self.datset_folds = None
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
        self.dataset_folds = Dataset.kfolds(self.dataset,n_fold)
        if self.verbose:
            print('split dataset into ',n_fold,' folds')
        return self


    def run(self):
        if self.dataset_folds is not None:
            self.model_folds = []
            for fold_id in range(self.n_fold):
                train_model = copy(self.model)
                train_model.load_dataset(self.dataset_folds[fold_id])
                train_model.fit()
                self.model_folds.append(train_model)
                if self.verbose:
                    print ('fold ',fold_id+1,' is trained')

    def score(self,metirc_func):
        score_matrix = numpy.zeros([self.n_fold,self.n_fold])
        for dataset_id in range(len(self.dataset_folds)):
            rating_list =  self.dataset_folds[dataset_id].ratings.to_list()    
            for model_id in range(len(self.model_folds)):
                predictions = self.model_folds[model_id].predict()
                score_matrix[dataset_id][model_id] = metirc_func(rating_list,predictions)
        if self.verbose:
            print('score table:')
            
            str_format=""
            for model_id in range(len(self.model_folds)):
                str_format += "%12.10s"
            for dataset_id in range(len(self.dataset_folds)):
                print(str_format % tuple(score_matrix[dataset_id]))
        return score_matrix









