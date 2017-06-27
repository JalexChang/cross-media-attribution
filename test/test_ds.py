import numpy
import unittest
from mta.ds.touch_row import TouchRow
from mta.ds.rating_row import RatingRow

class TestDs(unittest.TestCase):
    
    def setUp(self):
        self.ratin_rows = numpy.loadtxt('cust_ratings',delimiter=',')
        self.touch_rows = numpy.loadtxt('cust_touchs',delimiter=',')
    
    def test_rating_row(self):
        ratings = self.ratin_rows
        ratings = RatingRow(ratings,[max(ratings[:,0])+1,max(ratings[:,1])+1])
        row_list = ratings.to_list()
        matrix = ratings.to_matrix()
        self.assertEqual( (len(row_list),len(row_list[0])) , (13,3) , 'to_list()')
        self.assertEqual( (len(matrix),len(matrix[0])) , (5,5) , 'to_matrix()')
        self.assertEqual( ratings.mean() , self.ratin_rows[:,2].mean() , 'mean()')
        self.assertEqual( ratings.range(), 4 , 'range()')
        
    def test_touch_row(self):
        touchs = self.touch_rows
        touchs = TouchRow(touchs,[max(touchs[:,0])+1,max(touchs[:,1])+1])
        row_list = touchs.to_list()
        matrix = touchs.to_matrix()
        self.assertEqual( (len(row_list),len(row_list[0])) , (9,2) , 'to_list()')
        self.assertEqual( (len(matrix),len(matrix[0])) , (5,3) , 'test to_matrix()')


if __name__ == '__main__' :
    unittest.main()
    
    
