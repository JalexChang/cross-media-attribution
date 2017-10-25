from math import sqrt
import numpy

def rmse(rating_list,predicted_matrix):
    return sqrt(mse(rating_list,predicted_matrix))

def mse(rating_list,predicted_matrix):
    squared_error = 0.
    for u_id, i_id ,rating in rating_list:
        squared_error += pow(rating - predicted_matrix[u_id][i_id],2)
    mean_squared_error = squared_error / len(rating_list)
    return mean_squared_error

def mae(rating_list,predicted_matrix):
    abs_error = 0.
    for u_id, i_id ,rating in rating_list:
        abs_error += abs(rating - predicted_matrix[u_id][i_id]) 
    mean_abs_error = abs_error / len(rating_list)
    return mean_abs_error

def mape(rating_list,predicted_matrix):
    abs_percentage_error = 0.
    for u_id, i_id ,rating in rating_list:
        abs_percentage_error += abs((rating - predicted_matrix[u_id][i_id]) / rating) 
    mean_abs_percentage_error = abs_percentage_error / len(rating_list)
    return mean_abs_percentage_error * 100

def R2(rating_list, predicted_matrix):
    mean_rating =0.
    for u_id, i_id ,rating in rating_list:
        mean_rating+=rating
    mean_rating /= len(rating_list)

    sst = 0.
    ssr = 0.
    for u_id, i_id ,rating in rating_list:
        sst += pow(rating - mean_rating, 2)
        ssr += pow(rating -predicted_matrix[u_id][i_id], 2)
    return 1 - (ssr/sst)

def hit_rate_1(rating_list,predicted_matrix):
	return _hit_rate(rating_list,predicted_matrix,delta=0.1)

def hit_rate_2(rating_list,predicted_matrix):
	return _hit_rate(rating_list,predicted_matrix,delta=0.2)

def hit_rate_3(rating_list,predicted_matrix):
	return _hit_rate(rating_list,predicted_matrix,delta=0.3)

def _hit_rate(rating_list,predicted_matrix,delta=0.1):
    hits = 0
    for u_id, i_id ,rating in rating_list:
        abs_error = abs(rating - predicted_matrix[u_id][i_id])
        if abs_error <= delta:
            hits +=1
    mean_hits = float(hits) / len(rating_list)
    return mean_hits * 100

