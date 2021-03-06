import numpy
import statistics

class RatingRow:
    def __init__(self,rows,matrix_shape):
        self._user_ids = []
        self._item_ids = []
        self._ratings = [] 
        self._datatime =[]
        self.len = len(rows)
        for index in range(self.len):
            self._user_ids.append(int(rows[index][0]))
            self._item_ids.append(int(rows[index][1]))
            self._ratings.append(float(rows[index][2]))
            if len(rows[index]) ==4:
                self._datatime.append(rows[index][3])
        self._size_user = int(matrix_shape[0])
        self._size_item = int(matrix_shape[1])

    def to_list(self,with_datetime=False):
        rating_list = []
        for index in range(self.len):
            user_id = self._user_ids[index]
            item_id = self._item_ids[index]
            rating = self._ratings[index]
            if with_datetime is True and len(self._datatime) >0 :
                rating_list.append([user_id,item_id,rating, self._datatime[index]])
            else :
                rating_list.append([user_id,item_id,rating])
                
        return rating_list

    def to_matrix(self):
        matrix = numpy.zeros(self.matrix_shape())
        for index in range(self.len):
            user_id = self._user_ids[index]
            item_id = self._item_ids[index]
            rating = self._ratings[index]
            matrix[user_id][item_id] = rating
        return matrix

    def matrix_shape(self):
        return (self._size_user,self._size_item)

    def mean(self):
        return statistics.mean(self._ratings)

    def std(self):
        return statistics.stdev(self._ratings)

    def range(self):
        return max(self._ratings)-min(self._ratings)

