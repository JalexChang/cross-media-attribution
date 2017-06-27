import numpy

class TouchRow:
    def __init__(self,rows,matrix_shape):
        self._user_ids = []
        self._factor_ids = []
        self.len = len(rows)
        for index in range(self.len):
            self._user_ids.append(int(rows[index][0]))
            self._factor_ids.append(int(rows[index][1]))
        self._size_user = int(matrix_shape[0])
        self._size_factor = int(matrix_shape[1])
    
    def to_list(self):
        touch_list = []
        for index in range(self.len):
            user_id = self._user_ids[index]
            factor_id = self._factor_ids[index]
            touch_list.append([user_id,factor_id])
        return touch_list

    def to_matrix(self):
        matrix = numpy.zeros(self.matrix_shape())
        for index in range(self.len):
            user_id = self._user_ids[index]
            factor_id = self._factor_ids[index]
            matrix[user_id][factor_id] = 1
        return matrix

    def matrix_shape(self):
        return (self._size_user,self._size_factor)

    
