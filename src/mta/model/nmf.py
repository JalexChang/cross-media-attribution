import numpy 
from mta.model.mf import MF

class NMF(MF):
    def _calculate_updated_W(self,R_list,R_predicted):
        updated_W = numpy.copy(self.W)
        W_num = numpy.zeros([self._size_user,self._size_factor])   #numerators of W factors
        W_demon = numpy.zeros([self._size_user,self._size_factor]) #denominators of W factors
        number_element = numpy.zeros(self._size_user) # number of elements belonging to each n 
        
        for u_id, i_id, rating in R_list:
            for f_id in range(self._size_factor):
                W_num[u_id][f_id] += self.H[f_id][i_id]*rating
                W_demon[u_id][f_id] += self.H[f_id][i_id]*R_predicted[u_id][i_id]
                number_element[u_id] +=1  
        for u_id in range(self._size_user):
            for f_id in range(self._size_factor):
                if W_demon[u_id][f_id] >0 and self.W[u_id][f_id] >0:
                    W_demon[u_id][f_id] += number_element[u_id] * self.beta * self.W[u_id][f_id]
                    updated_W[u_id][f_id] *= W_num[u_id][f_id] / W_demon[u_id][f_id]
        return updated_W

    def _calculate_updated_H(self,R_list,R_predicted):
        updated_H = numpy.copy(self.H)
        H_num = numpy.zeros([self._size_factor,self._size_item])   #numerators of H factors
        H_demon = numpy.zeros([self._size_factor,self._size_item]) #denominators of H factors
        number_element = numpy.zeros(self._size_item) # number of elements belonging to each m
        
        for u_id, i_id, rating in R_list:
            for f_id in range(self._size_factor):
                H_num[f_id][i_id] += self.W[u_id][f_id]*rating
                H_demon[f_id][i_id] += self.W[u_id][f_id]*R_predicted[u_id][i_id]
                number_element[i_id] +=1
        for u_id in range(self._size_item):
            for f_id in range(self._size_factor):
                if H_demon[f_id][u_id]>0:
                    H_demon[f_id][u_id] += number_element[u_id] * self.beta * self.H[f_id][u_id]
                    updated_H[f_id][u_id] *= H_num[f_id][u_id] / H_demon[f_id][u_id]
        return updated_H