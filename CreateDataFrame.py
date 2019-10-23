from __future__ import division
import pandas
import pandas as pd
from TreeFunctions import *
import numpy as np
from scipy.linalg import fractional_matrix_power
from general import *
import math
from Contact import *

def lookUpPCAByID(id, dataFrame, componentNumber):
	dataFrame = dataFrame[componentNumber]
	try:
		values = dataFrame.loc[id]
		return values
	except:
		return "?"

def assignPCATipValuesByLanguageID(tree, dataFrame, componentNumber, checkInternalNodes = True):
	outputTree = tree.copy()
	tips = findTips(outputTree)
	for tip in tips:
		outputTree[tip] = {}
		id = findGlottologId(tip)
		value = lookUpPCAByID(id, dataFrame, componentNumber)
		outputTree[tip] = {}
		outputTree[tip]['state'] = value
	if checkInternalNodes:
		nodes = outputTree.keys()
		done = False
		while done == False:
			restart = False
			nodes = outputTree.keys()
			if nodes == []:
				done = True
			tips = findTips(outputTree)
			for node in nodes:
				id = findGlottologId(node)
				value = lookUpPCAByID(id, dataFrame, componentNumber)	
				if not value == '?':
					descendants = findDescendantNodes(node)
					if len(descendants) > 0:
						newName = findNodeNameWithoutStructure(node)
						outputTree = dropDescendantNodes(outputTree, node, newName)
						restart = True
						break
					else:
						if outputTree[node] == UNASSIGNED:
							outputTree[node] = {}
							outputTree[node]['state'] = value			
			if not restart:
				done = True
	return outputTree

def createPCATrees(treeFile, dataFrame, componentNumber):
	trees = open(treeFile).readlines()
	for m in xrange(len(trees)):
		tree = trees[m]
		tree = tree.strip('\n')
		tree = createTree(tree)	
		tree = assignPCATipValuesByLanguageID(tree, dataFrame, componentNumber, checkInternalNodes = True)
		trees[m] = tree
	return trees
	
def createSimplePCADataFrame(dataFrame, trees, componentNumber, howFarBack = 1, threshold = 500, limit = 5, languagesFile = 'languages.txt', excludeRelatives = True):
	languages = pd.read_csv(languagesFile, header = 0, sep =',')	
	languages['index'] = languages['ID']
	languages = languages.set_index('index')
	languageLocations = arrangeLanguagesByLongitudeAndLatitude(languages['ID'], languages['Longitude'], languages['Latitude'])
	rowsList = []
	for tree in trees:
		tips = findTips(tree)
		for tip in tips:
			id = findGlottologId(tip)
			tipState = tree[tip]['state']
			if not tipState == '?':
				try:
					relatives = findRelatives(tree, tip, howFarBack)
					relativesValue = 0
					relativesNumber = 0
					for relative in relatives:
						relativeId = findGlottologId(relative)
						try:
							state = dataFrame.loc[relativeId, componentNumber]
							if not state == '?':
								relativesValue = relativesValue + state
								relativesNumber = relativesNumber + 1
						except:
							pass
					if relativesNumber > 0:
						relativesValue = relativesValue/relativesNumber
					else:
						relativesValue = None
				except:
					relativesValue = None
					relatives = []
# 				this could fail, but shouldn't if all languages are there in languages.txt
				id = findGlottologId(tip)					
				lat = languages.loc[id, 'Latitude']
				long = languages.loc[id, 'Longitude']
				if excludeRelatives:
					toExclude = relatives
					toExclude = [findGlottologId(x) for x in toExclude]
				else:
					toExclude = None
# 				print id
# 				print lat
# 				print math.isnan(lat)
				if not math.isnan(lat) and not math.isnan(long):
					neighbours = findNeighbours(lat, long, languageLocations, threshold, limit, toExclude = toExclude)
# 					print 'step 2a'
					neighbourTotal = 0
					numberOfNeighbours = 0
					for neighbour in neighbours:
						try:
# 							print 'step 2b'
# 							print neighbour[0]
# 							print componentNumber
							value = dataFrame.loc[neighbour[0], componentNumber]
# 							print value
# 							print 'step 2c'
							if not value == '?':
								neighbourTotal = neighbourTotal + value
								numberOfNeighbours = numberOfNeighbours + 1 
						except:
							pass
					if numberOfNeighbours > 0:
						neighbourValue = neighbourTotal / numberOfNeighbours
					if numberOfNeighbours == 0:
						neighbourValue = None
				else:
					neighbourValue = None
				row = {}
				row['tip'] = tip
				row['TipState'] = tipState
				row['RelativesValue'] = relativesValue
				row['NeighbourValue'] = neighbourValue					
				rowsList.append(row)			

	resultDataFrame = pandas.DataFrame(rowsList)
	return resultDataFrame



