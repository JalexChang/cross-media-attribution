import numpy
import math
from mta.dataset import Dataset
from copy import copy

class Profile:
    @classmethod
    def rating_dist(self, dataset, stacks=11, stack_range=1000):
        dist = numpy.zeros(stacks, dtype=int)
        labels = numpy.array(list( i*stack_range for i in range(stacks)))
        for u_id, i_id, rating in dataset.ratings.to_list():
            if rating >= stacks* stack_range:
                dist[stacks-1] +=1
            else:
                index = math.floor(rating/stack_range)
                dist[index] +=1
        return dist, labels

    @classmethod
    def rating_std_dist_on_items(self, dataset, stacks=11, stack_range=100):
        item_size = dataset.ratings.matrix_shape()[1]
        rating_stds  = numpy.zeros(item_size, dtype=float) 
        rating_matrix = dataset.ratings.to_matrix()
        dist = numpy.zeros(stacks, dtype=int)
        labels = numpy.array(list( i*stack_range for i in range(stacks)))
        
        for i_id in range(len(rating_matrix[0])):
            rating_stds[i_id] = rating_matrix[:,i_id].std(ddof=1)
        
        for std_value in rating_stds:
            if std_value >= stacks* stack_range:
                dist[stacks-1]+=1
            else:
                index = math.floor(std_value/stack_range)
                dist[index]+=1
        return dist, labels

    #get the distribution of number of touched factors in each users
    @classmethod
    def factor_dist_on_users(self, dataset, stacks=21, stack_range=1):
        user_size = dataset.matrix_shape()[0]
        factor_size = dataset.matrix_shape()[1]
        user_num_factors = numpy.zeros(user_size, dtype=int)
        dist = numpy.zeros(stacks, dtype=int)
        labels = numpy.array(list( i*stack_range for i in range(stacks)))

        user_factor_matrix = dataset.touchs.to_matrix()
        for u_id in range(user_size):
            user_num_factors[u_id] = user_factor_matrix[u_id].sum()
        for num_factor in user_num_factors:
            if num_factor >= stacks*stack_range:
                dist[stacks-1] +=1
            else :
                index = math.floor(num_factor/stack_range)
                dist[index] +=1
        return dist, labels

    #get the distribution of number of touched factors in each items 
    @classmethod
    def factor_dist_on_items(self, dataset,  stacks=21, stack_range=1):
        item_size = dataset.matrix_shape()[2]
        factor_size = dataset.matrix_shape()[1]
        item_factor_table  = numpy.zeros((item_size, factor_size), dtype=bool)
        item_num_factors = numpy.zeros(item_size, dtype=int)
        dist = numpy.zeros(stacks, dtype=int)
        labels = numpy.array(list( i*stack_range for i in range(stacks)))
        
        user_factor_matrix = dataset.touchs.to_matrix()
        for u_id, i_id, rating in dataset.ratings.to_list():
            for f_id in range(factor_size):
                if user_factor_matrix[u_id][f_id] ==1:
                    item_factor_table[i_id][f_id] =True
        for i_id in range(len(item_factor_table)):
            item_num_factors[i_id] = item_factor_table[i_id].sum()

        for num_factor in item_num_factors:
            if num_factor >= stacks*stack_range:
                dist[stacks-1] +=1
            else :
                index = math.floor(num_factor/stack_range)
                dist[index] +=1
        return dist, labels

    #get the distribution of number of products in each factors
    @classmethod
    def item_dist(self, dataset, stacks=11, stack_range=10):
        matrix_shape = dataset.ratings.matrix_shape()
        items  = numpy.zeros(matrix_shape[1], dtype=int)
        
        for u_id, i_id, rating in dataset.ratings.to_list():
            items[i_id] +=1
        dist= numpy.zeros(stacks, dtype=int)
        for item_size in items:
            if item_size >= stacks* stack_range:
                dist[stacks-1]+=1
            else:
                index = math.floor(item_size/stack_range)
                dist[index]+=1
        return dist

    #get the distribution of number of products in each factors
    @classmethod
    def item_dist_on_factors(self, dataset, stacks=11, stack_range=10):
        matrix_shape = ( dataset.matrix_shape()[1],dataset.matrix_shape()[2])
        factor_item_table  = numpy.zeros(matrix_shape, dtype=bool)
        factor_num_items = numpy.zeros(matrix_shape[0], dtype=int)
        dist = numpy.zeros(matrix_shape[1], dtype=int)
        user_factor_matrix = dataset.touchs.to_matrix()

        for u_id, i_id, rating in dataset.ratings.to_list():
            for f_id in range(matrix_shape[0]):
                if user_factor_matrix[u_id][f_id] ==1:
                    factor_item_table[f_id][i_id] =True

        for f_id in range(len(factor_item_table)):
            factor_num_items[f_id] = factor_item_table[f_id].sum()

        for num_item in factor_num_items:
            dist[num_item] +=1

        for index in reversed(range(len(dist))):
            if dist[index] >0:
                dist = dist[:index+1]
                break
        return dist

