from __future__ import division
import pandas
from TreeFunctions import *
import numpy as np
from scipy.linalg import fractional_matrix_power
from general import *
from math import sin, cos, asin, sqrt
import random

def arrangeLanguagesByLongitudeAndLatitude(languages, longitudes, latitudes):
	dictionary = {}
	for i in xrange(len(languages)):
		language = languages[i]
		lat = latitudes[i]
		long = longitudes[i]
		try:
			key = str(int(lat)) + '_' + str(int(long))
			try:
				dictionary[key].append((language, lat, long))
			except:
				dictionary[key] = []
				dictionary[key].append((language, lat, long))
		except:
			pass
	return dictionary

def findLanguagesInRegion(lat1, lat2, long1, long2, languageLocations):
	 lat1 = int(lat1)
	 lat2 = int(lat2) + 1
	 long1 = int(long1)
	 long2 = int(long2) + 1
	 result = []
	 for x in range(lat1, lat2 + 1):
	 	for y in range(long1, long2 + 1):
	 		try:
	 			temp = languageLocations[str(x) + '_' + str(y)]
	 			result = result + temp
	 		except:
	 			pass
	 return result
	
def findNeighbours(lat, long, languageLocations, threshold, limit, toExclude = None):
	lat1 = lat - limit
	lat2 = lat + limit
	long1 = long - limit
	long2 = long + limit
	languagesInRegion = findLanguagesInRegion(lat1, lat2, long1, long2, languageLocations)
	lon1 = long
	lat1 = lat
	result = []
	for language in languagesInRegion:
		lat2 = language[1]
		lon2 = language[2]
		distance = haversine(lon1, lat1, lon2, lat2)		
		if distance < threshold:
			if not toExclude == None:
				if not language[0] in toExclude:
					result.append(language)
			else:
				result.append(language)
	return result
	