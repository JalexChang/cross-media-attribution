import numpy
import math
from mta.dataset import Dataset
from copy import copy

class Profile:
    @classmethod
    def ratings_dist(self, ratings, ticks=10):
        max_element = max(ratings.to_matrix().flat)
        tick_range = max_element / (ticks-1) if ticks >=1 else 1
        dist = numpy.zeros(ticks)
        labels = numpy.array(list( i*tick_range for i in range(ticks)))
        for u_id, i_id, rating in ratings.to_list():
            index = math.floor(rating/tick_range)
            dist[index] +=1
        return dist, labels
'''
    @classmethod
    def touchs_dist(self, dataset, ticks=10):
        touchs = dataset.ratings
        max_element = max(touchs.to_matrix().flat)
        tick_range = max_element / (ticks-1) if ticks >=1 else 1
        dist = numpy.zeros(ticks)
        for u_id, f_id in touchs.to_list():
            index = math.floor(rating/tick_range)
            dist[index] +=1
        return dist
'''

