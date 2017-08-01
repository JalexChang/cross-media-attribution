import numpy
import math
from mta.dataset import Dataset
from copy import copy

class Profile:
    @classmethod
    def ratings_dist(self, ratings, stacks=10):
        max_element = max(ratings.to_matrix().flat)
        stack_range = max_element / (stacks-1) if stacks >=1 else 1
        dist = numpy.zeros(stacks, dtype=int)
        labels = numpy.array(list( i*stack_range for i in range(stacks)))
        for u_id, i_id, rating in ratings.to_list():
            index = math.floor(rating/stack_range)
            dist[index] +=1
        return dist, labels

    @classmethod
    def item_dist(self, ratings, stacks=11, stack_range=10):
        matrix_shape = ratings.matrix_shape()
        items  = numpy.zeros(matrix_shape[1], dtype=int)
        for u_id, i_id, rating in ratings.to_list():
            items[i_id] +=1
        dist= numpy.zeros(stacks, dtype=int)
        for item_size in items:
            if item_size >= stacks* stack_range:
                dist[stacks-1]+=1
            else:
                index = math.floor(item_size/stack_range)
                dist[index]+=1
        return dist

    @classmethod
    def factor_size_dist(self, touchs):
        matrix_shape = touchs.matrix_shape()
        user_factors  = numpy.zeros(matrix_shape[0], dtype=int)
        dist = numpy.zeros(matrix_shape[1], dtype=int)
        for u_id, f_id in touchs.to_list():
            user_factors[u_id] +=1
        for factor_size in user_factors:
            dist[factor_size] +=1
        for index in reversed(range(len(dist))):
            if dist[index] >0:
                dist = dist[:index+1]
                break
        return dist

