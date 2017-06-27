import numpy

class RatingRow:
    def __init__(self,rows,matrix_shape):
        self._user_ids = []
        self._item_ids = []
        self._ratings = [] 
        self.len = len(rows)
        for index in range(self.len):
            self._user_ids.append(int(rows[index][0]))
            self._item_ids.append(int(rows[index][1]))
            self._ratings.append(rows[index][2])
        self._size_user = int(matrix_shape[0])
        self._size_item = int(matrix_shape[1])

    def to_list(self):
        rating_list = []
        for index in range(self.len):
            user_id = self._user_ids[index]
            item_id = self._item_ids[index]
            rating = self._ratings[index]
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
        return sum(self._ratings)/self.len

    def range(self):
        return max(self._ratings)-min(self._ratings)

