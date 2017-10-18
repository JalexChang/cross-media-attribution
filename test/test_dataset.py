import numpy
import unittest
from mta.dataset import Dataset
import mta.utility as utility
from datetime import timedelta


class TestDataSet(unittest.TestCase):

    def setUp(self):
        self.ratings = numpy.loadtxt('cust_ratings',delimiter=',')
        self.touchs = numpy.loadtxt('cust_touchs',delimiter=',')
        self.dataset = Dataset(self.ratings,self.touchs)

    def test_init(self):
        self.assertIsInstance(self.dataset,Dataset)

    def test_invalid_data_format(self):
        with self.assertRaises(Exception):
            new_dataset = Dataset(ratings,ratings)
        with self.assertRaises(Exception):
            new_dataset = Dataset(touchs,touchs) 

    def test_matrix_shape(self):
        matrix_shape = self.dataset.matrix_shape()
        self.assertEqual(matrix_shape, (5,3,5))

    def test_density(self):
        density = self.dataset.density()
        self.assertEqual(density,0.52)
 
    def test_init_with_martix_shape(self):
        new_ratings = numpy.copy(self.dataset.ratings.to_list())
        new_touchs = numpy.copy(self.dataset.touchs.to_list())
        new_matrix_shape = self.dataset.matrix_shape()
        new_dataset = Dataset(new_ratings,new_touchs,new_matrix_shape)
        self.assertIsInstance(new_dataset,Dataset)

    def test_seperate_rating_item_by_item(self):
        rating_set , touch_set = self.dataset.seperate_rating_touch_by_item()
        self.assertEqual(len(rating_set), len(touch_set))
        for i_id in range(len(rating_set)):
            self.assertEqual(len(rating_set[i_id]), len(touch_set[i_id]))

    def test_train_test_split(self):
        train_dataset,test_dataset = Dataset.train_test_split(self.dataset,test_size=0.5)
        self.assertEqual(train_dataset.matrix_shape(),test_dataset.matrix_shape())
        size_rating_train = len(train_dataset.ratings.to_list())
        size_rating_test = len(test_dataset.ratings.to_list())
        self.assertEqual(size_rating_train+size_rating_test , len(self.dataset.ratings.to_list()))
        self.assertEqual(train_dataset.touchs.matrix_shape(),test_dataset.touchs.matrix_shape())

    def test_kfolds(self):
        dataset_folds = Dataset.kfolds(self.dataset,n_folds=5)
        self.assertEqual(len(dataset_folds),5)
        for dataset_fold in dataset_folds:
            self.assertEqual(dataset_fold.matrix_shape(),self.dataset.matrix_shape())
    
    def test_split_by_time_range(self):
        ratings = utility.load_rating_file("cust_ratings_with_datetime")
        dataset = Dataset(ratings, self.touchs)
        dataset_folds = Dataset.split_by_time_range(dataset,time_range=timedelta(1))
        self.assertEqual(len(dataset_folds),5)
        for dataset_fold in dataset_folds:
            self.assertEqual(dataset_fold.matrix_shape(),dataset.matrix_shape())

    def test_split_by_time_range_exception(self):
        with self.assertRaises(Exception):
            dataset_folds = Dataset.split_by_time_range(self.dataset,time_range=timedelta(1))

    def test_append(self):
        new_dataset = Dataset.append(self.dataset,self.dataset)
        self.assertEqual(new_dataset.matrix_shape(),self.dataset.matrix_shape())
        self.assertEqual(len(new_dataset.ratings.to_list()), 2*len(self.dataset.ratings.to_list()))

    def test_merge(self):
        datasets = [self.dataset,self.dataset,self.dataset]
        new_dataset = Dataset.merge(datasets)
        self.assertEqual(new_dataset.matrix_shape(), self.dataset.matrix_shape())
        self.assertEqual(len(new_dataset.ratings.to_list()), 3*len(self.dataset.ratings.to_list()))


if __name__ == '__main__' :
    unittest.main()
    
    
