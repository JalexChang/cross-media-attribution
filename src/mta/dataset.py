import numpy
import math
from mta.ds.touch_row import TouchRow
from mta.ds.rating_row import RatingRow
from datetime import datetime
from datetime import timedelta
from copy import copy


'''
    formats of source data is:
    user_id, item_id, ratings

'''
class Dataset:
    def __init__ (self,rating_rows,touch_rows,matrix_shape=None):
        self._construct_matrix_shape(rating_rows,touch_rows,matrix_shape)
        self._construct_ratings(rows=rating_rows)
        self._construct_touchs(rows=touch_rows)

    def _construct_matrix_shape(self,rating_rows,touch_rows,matrix_shape=None):
        if matrix_shape is None:
            self._size_user = int(max(list( int(rating[0]) for rating in rating_rows )))+1
            self._size_item = int(max(list( int(rating[1]) for rating in rating_rows )))+1
            self._size_factor = int(max(list( int(touch[1]) for touch in touch_rows )))+1
        else:
            self._size_user = matrix_shape[0]
            self._size_factor = matrix_shape[1]
            self._size_item = matrix_shape[2]

    def _construct_ratings(self,rows):
        if len(rows[0]) <3 :
            raise Exception('invalid data format when construct ratings') 
        self.ratings = RatingRow(rows,(self._size_user,self._size_item))
        
    def _construct_touchs(self,rows):
        if len(rows[0]) !=2 :
            raise Exception('invalid data format when construct touchs') 
        self.touchs = TouchRow(rows, (self._size_user, self._size_factor))

    def matrix_shape(self):
        return (self._size_user,self._size_factor,self._size_item)
    
    def density(self):
        return self.ratings.len / (self._size_user * self._size_item)

    #convert dataset into reresion datasets seperated by item_id  
    def seperate_rating_touch_by_item(self):
        size_item = self.ratings.matrix_shape()[1]
        touch_matrix = self.touchs.to_matrix()
        rating_set = []
        touch_set= []
        for i_id in range(size_item):
            rating_set.append([])
            touch_set.append([])

        for u_id, i_id, rating in self.ratings.to_list():
            rating_set[i_id].append(rating)
            row_touch = touch_matrix[u_id].tolist()
            touch_set[i_id].append(row_touch)
        return rating_set, touch_set

    @classmethod
    def train_test_split(self,dataset,test_size=0.2):
        matrix_shape = dataset.matrix_shape()
        touchs = numpy.copy(dataset.touchs.to_list())
        ratings= numpy.copy(dataset.ratings.to_list())
        numpy.random.shuffle(ratings)
        split_index = int(len(ratings)*test_size)+1
        test_ratings = ratings[:split_index]
        train_ratings = ratings[split_index:]
        train_dataset = Dataset(train_ratings,touchs,matrix_shape)
        test_dataset = Dataset(test_ratings,touchs,matrix_shape)
        return train_dataset,test_dataset

    @classmethod
    def kfolds(slef,dataset,n_folds=5):
        matrix_shape = dataset.matrix_shape()
        touchs = numpy.copy(dataset.touchs.to_list())
        ratings= numpy.copy(dataset.ratings.to_list())
        numpy.random.shuffle(ratings)
        rating_folds = numpy.array_split(ratings,n_folds)
        dataset_folds = []
        for rating_fold in rating_folds:
            dataset_fold = Dataset(rating_fold,touchs,matrix_shape)
            dataset_folds.append(dataset_fold)
        return dataset_folds

    @classmethod
    def split_by_time_range(slef,dataset,time_range=timedelta(days=7)):
        ratings = dataset.ratings.to_list(with_datetime=True)
        if len(ratings[0]) < 4 :
            raise Exception("the dataset does not contain datetime")
        touchs = dataset.touchs.to_list()
        matrix_shape = dataset.matrix_shape()
        time_base = ratings[0][3] + time_range
        ratings_set = []
        datasets = []
        low_bound =0
        
        for high_bound  in range(len(ratings)):
            now_date = ratings[high_bound][3]
            if now_date >= time_base:
                ratings_set.append(ratings[low_bound : high_bound])
                low_bound = high_bound
                time_base+= time_range
        if low_bound <= len(ratings)-1:
            ratings_set.append(ratings[low_bound:len(ratings)])

        for rating in ratings_set:
            datasets.append(Dataset(rating,touchs,matrix_shape))
        return datasets

    @classmethod
    def append(self,dataset_src1, dataset_src2):
        if dataset_src1.matrix_shape() != dataset_src2.matrix_shape():
            raise Exception("datasets' matrix shapes are not match")
        matrix_shape = dataset_src1.matrix_shape()
        touchs = numpy.copy(dataset_src1.touchs.to_list())
        ratings = numpy.vstack((dataset_src1.ratings.to_list(), dataset_src2.ratings.to_list()))
        return Dataset(ratings,touchs,matrix_shape)

    @classmethod
    def merge(self,datasets):
        if len(datasets) <1 :
            raise Exception('length of datasets is less than 2')
        matrix_shape = datasets[0].matrix_shape()
        touchs = numpy.copy(datasets[0].touchs.to_list())
        ratings = numpy.copy(datasets[0].ratings.to_list())
        for dataset in datasets[1:]:
            ratings = numpy.vstack((ratings, dataset.ratings.to_list()))
        return Dataset(ratings, touchs, matrix_shape)















