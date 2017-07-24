import numpy
from mta.ds.touch_row import TouchRow
from mta.ds.rating_row import RatingRow
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
            if max(rating_rows[:,0]) != max(touch_rows[:,0]):
                raise Exception('different number of users in ratings and touchs')
            self._size_user = int(max(rating_rows[:,0]))+1
            self._size_item = int(max(rating_rows[:,1]))+1
            self._size_factor = int(max(touch_rows[:,1]))+1
        else:
            self._size_user = matrix_shape[0]
            self._size_factor = matrix_shape[1]
            self._size_item = matrix_shape[2]

    def _construct_ratings(self,rows):
        if len(rows[0]) !=3 :
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















