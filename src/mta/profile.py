import numpy
import math
from mta.dataset import Dataset
from copy import copy

class Profile:
    @classmethod
    def rating_dist(self, dataset, stacks=11, stack_range=1000):
        rating_list = numpy.array(dataset.ratings.to_list())[:,2].flat
        
        dist = self._make_dist(rating_list, stacks, stack_range)
        labels = self._make_labels(stacks, stack_range)
        return dist, labels

    @classmethod
    def rating_std_dist_on_items(self, dataset, stacks=11, stack_range=100):
        item_size = dataset.ratings.matrix_shape()[1]
        rating_stds  = numpy.zeros(item_size, dtype=float) 
        rating_matrix = dataset.ratings.to_matrix()    
        for i_id in range(len(rating_matrix[0])):
            rating_stds[i_id] = rating_matrix[:,i_id].std(ddof=1)
        
        dist = self._make_dist(rating_stds, stacks, stack_range)
        labels = self._make_labels(stacks, stack_range)
        return dist, labels

    #get the distribution of number of touched factors in each users
    @classmethod
    def factor_dist_on_users(self, dataset, stacks=21, stack_range=1):
        user_size = dataset.matrix_shape()[0]
        factor_size = dataset.matrix_shape()[1]
        user_num_factors = numpy.zeros(user_size, dtype=int)
        user_factor_matrix = dataset.touchs.to_matrix()
        for u_id in range(user_size):
            user_num_factors[u_id] = user_factor_matrix[u_id].sum()

        dist = self._make_dist(user_num_factors, stacks, stack_range)
        labels = self._make_labels(stacks, stack_range)
        return dist, labels

    #get the distribution of the number of touched factor in each items 
    @classmethod
    def factor_dist_on_items(self, dataset,  stacks=21, stack_range=1):
        item_size = dataset.matrix_shape()[2]
        factor_size = dataset.matrix_shape()[1]
        item_factor_table  = numpy.zeros((item_size, factor_size), dtype=bool)
        item_num_factors = numpy.zeros(item_size, dtype=int)
        user_factor_matrix = dataset.touchs.to_matrix()
        for u_id, i_id, rating in dataset.ratings.to_list():
            for f_id in range(factor_size):
                if user_factor_matrix[u_id][f_id] ==1:
                    item_factor_table[i_id][f_id] =True
        for i_id in range(len(item_factor_table)):
            item_num_factors[i_id] = item_factor_table[i_id].sum()

        dist = self._make_dist(item_num_factors, stacks, stack_range)
        labels = self._make_labels(stacks, stack_range)
        return dist, labels

    @classmethod
    def item_dist(self, dataset, stacks=11, stack_range=10):
        item_size = dataset.matrix_shape()[2]
        items  = numpy.zeros(item_size, dtype=int)
        for u_id, i_id, rating in dataset.ratings.to_list():
            items[i_id] +=1

        dist = self._make_dist(items, stacks, stack_range)
        labels = self._make_labels(stacks, stack_range)
        return dist, labels

    #get the distribution of the number of item in each factors
    @classmethod
    def item_dist_on_factors(self, dataset, stacks=11, stack_range=10):
        item_size = dataset.matrix_shape()[2]
        factor_size = dataset.matrix_shape()[1]
        factor_item_table  = numpy.zeros((factor_size, item_size), dtype=bool)
        factor_num_items = numpy.zeros(factor_size, dtype=int)
        user_factor_matrix = dataset.touchs.to_matrix()
        for u_id, i_id, rating in dataset.ratings.to_list():
            for f_id in range(factor_size):
                if user_factor_matrix[u_id][f_id] ==1:
                    factor_item_table[f_id][i_id] =True
        for f_id in range(len(factor_item_table)):
            factor_num_items[f_id] = factor_item_table[f_id].sum()

        dist = self._make_dist(factor_num_items, stacks, stack_range)
        labels = self._make_labels(stacks, stack_range)
        return dist, labels

    #get the distribution of the number of item in each users
    @classmethod
    def item_dist_on_users(self, dataset, stacks=11, stack_range=10):
        user_size = dataset.matrix_shape()[0]
        user_num_items = numpy.zeros(user_size, dtype=int)
        for u_id, i_id, rating in dataset.ratings.to_list():
            user_num_items[u_id]+=1
        dist = self._make_dist(user_num_items, stacks, stack_range)
        labels = self._make_labels(stacks, stack_range)
        return dist, labels
        
    def _make_dist(value_list, stacks, stack_range ):
        dist = numpy.zeros(stacks, dtype=int)
        for value in value_list:
            if value >= stacks* stack_range:
                dist[stacks-1] +=1
            else:
                index = math.floor(value/stack_range)
                dist[index] +=1
        return dist

    def _make_labels(stacks, stack_range):
        return numpy.array(list( i*stack_range for i in range(stacks)))
