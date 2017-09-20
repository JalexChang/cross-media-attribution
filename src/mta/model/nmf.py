import numpy 
from mta.model.mf import MF

class NMF(MF):
    def _init_latent_factors(self):
        if not self.trained:
            self.W = numpy.zeros([self._size_user,self._size_factor])
            for u_id,f_id in self.touchs.to_list():
                self.W[u_id][f_id] = numpy.random.random()
            self.H = numpy.random.random([self._size_factor,self._size_item])
            if self.verbose:
                print('latent factors has been initialized')

    def _update_sgd(self,R_list):
        W_num = numpy.zeros([self._size_user,self._size_factor])   #numerators of W factors
        W_demon = numpy.zeros([self._size_user,self._size_factor]) #denominators of W factors
        W_elements = numpy.zeros(self._size_user) # number of elements belonging to each n
        H_num = numpy.zeros([self._size_factor,self._size_item])   #numerators of H factors
        H_demon = numpy.zeros([self._size_factor,self._size_item]) #denominators of H factors
        H_elements = numpy.zeros(self._size_item) # number of elements belonging to each m
        for u_id, i_id, rating in R_list:
            predicted_rating = self._predict_one_element(u_id, i_id, p_type = "normal")
            #caculate numerators and denominators of factors
            for f_id in range(self._size_factor):
                if self.W[u_id][f_id] > 0 :
                    W_num[u_id][f_id] += self.H[f_id][i_id] * rating
                    W_demon[u_id][f_id] += self.H[f_id][i_id] * predicted_rating
                    W_elements[u_id] +=1
                    H_num[f_id][i_id] += self.W[u_id][f_id] * rating
                    H_demon[f_id][i_id] += self.W[u_id][f_id] * predicted_rating
                    H_elements[i_id] +=1
            #update biases
            error = rating - predicted_rating
            if self.item_biased: 
                self.bias_item[i_id] += self.alpha *( error - self.beta * self.bias_item[i_id])
            if self.user_biased:
                self.bias_user[u_id] += self.alpha *( error - self.beta * self.bias_user[u_id])

        #updated factors
        for u_id in range(self._size_user):
            for f_id in range(self._size_factor):
                if self.W[u_id][f_id] >0 and W_demon[u_id][f_id] >0 :
                    self.W[u_id][f_id] *= W_num[u_id][f_id] / (W_demon[u_id][f_id] + W_elements[u_id] * self.beta * self.W[u_id][f_id])
        for i_id in range(self._size_item):
            for f_id in range(self._size_factor):
                if H_demon[f_id][i_id]>0:
                    self.H[f_id][i_id] *= H_num[f_id][i_id] / (H_demon[f_id][i_id] + H_elements[i_id] * self.beta * self.H[f_id][i_id])
    
    