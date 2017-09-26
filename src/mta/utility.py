import numpy
import math
from datetime import datetime
from datetime import timedelta
from copy import copy


def load_rating_file(file_path, time_format="%Y-%m-%d"):
	ratings = []
	with open(file_path,"r") as f:
		for line in f.readlines():
			rating = line.rstrip().split(",")
			rating[0] = int(rating[0])
			rating[1] = int(rating[1])
			rating[2] = float(rating[2])
			if len(rating)==4 :
				rating[3] = datetime.strptime(rating[3],time_format)
			ratings.append(rating)
	return ratings

def load_touch_file(file_path):
	touchs = []
	with open(file_path,"r") as f:
		for line in f.readlines():
			touch = line.rstrip().split(",")
			touch[0] = int(touch[0])
			touch[1] = int(touch[1])
			touchs.append(touch)
	return touchs

class Normalization:
	
	def __init__(self):
		self.min_element = 0.
		self.max_element = 0.

	def log(self, R_list):
		for index in range(len(R_list)):
			R_list[index][2] = math.log(R_list[index][2])
		return self.minmax(R_list)

	def log_revert(self, R_list):
		R_list = self.minmax_revert(R_list)
		for index in range(len(R_list)):
			R_list[index][2] = math.exp(R_list[index][2])
		return R_list

	def minmax(self, R_list):
		self.min_element = min(list(float(rating[2]) for rating in R_list))
		self.max_element = max(list(float(rating[2]) for rating in R_list))
		for index in range(len(R_list)):
			R_list[index][2] = (R_list[index][2] - self.min_element) / (self.max_element - self.min_element)
		return R_list

	def minmax_revert(self, R_list):
		for index in range(len(R_list)):
			R_list[index][2] = (R_list[index][2] * (self.max_element - self.min_element)) + self.min_element
		return R_list
