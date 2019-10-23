from __future__ import division
import pandas
from TreeFunctions import *
from PrepareGrambankData import *
import numpy as np
from scipy.linalg import fractional_matrix_power
from general import *
from LikelihoodFunction import * 
from Reconstruction import *
from sklearn.linear_model import LinearRegression, Lasso
from CreateDataFrame import *
import tensorflow as tf


def mainAnalysis(data, trees, howFarBack, threshold, limit):
	dataFrame = createMainDataFrame(data, trees, howFarBack, threshold, limit)
	dataFrame = dataFrame.dropna()
	y = np.array(dataFrame['TipState'])
	x1 = np.array(dataFrame['AncestorState'])
	x2 = np.array(dataFrame['NeighbourValue'])
	X = np.transpose(np.array([x1, x2]))
	model = Lasso(alpha = 0.0001, fit_intercept = True, positive = True)
	model = model.fit(X, y)
	return model.coef_, model.intercept_

def tensorflowAnalysis(data, trees, howFarBack, threshold, limit):
    steps = 10000
    learn_rate = 0.03
    dataFrame = createMainDataFrame(data, trees, howFarBack, threshold, limit)
    dataFrame = dataFrame.dropna()
    target = np.array(dataFrame['TipState'])
    target = [[member] for member in target]
    x1 = np.array(dataFrame['AncestorState'])
    x2 = np.array(dataFrame['NeighbourValue'])
    X = np.transpose(np.array([x1, x2]))
    dependent_variables = np.array(X)	
    x = tf.placeholder(tf.float32, [None, 2], name = "x")
    W = tf.Variable(tf.zeros([2, 1]), name = "W")
    b1 = tf.Variable(tf.ones([1]), name = "b1")
    b2 = tf.Variable(tf.ones([1]), name = "b2")
    b = b1 * b2
#     b = tf.Variable(tf.zeros([1]), name = "b")
    y = tf.matmul(x, W) + b
#     y = tf.matmul(x, W)
    y_ = tf.placeholder(tf.float32, [None, 1])
    cost = tf.reduce_mean(tf.square(y_ - y))
    cost_sum = tf.summary.scalar("cost", cost)
    train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)	
    clip_op = tf.assign(W, tf.clip_by_value(W, 0, np.infty))
    clip_op2 = tf.assign(b1, tf.clip_by_value(b1, 0, np.infty))
    clip_op3 = tf.assign(b2, tf.clip_by_value(b2, 0, np.infty))
    sum = tf.reduce_sum(W) + b1
#     sum = tf.reduce_sum(W)
    normalise1 = tf.assign(W, W / sum)
    normalise2 = tf.assign(b1, b1 / sum)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in xrange(steps):
        feed = {x: X, y_: target}
        sess.run(train_step, feed_dict = feed)
        sess.run(clip_op)
        sess.run(clip_op2)
        sess.run(clip_op3)
#         print sess.run(b1)
#         print sess.run(b2)
#         print sess.run(b)
        
#         if sess.run(sum) > 1:
        sess.run(normalise1)
        sess.run(normalise2)
    coefficients = sess.run(W)
    return str(coefficients[0][0]) + ',' + str(coefficients[1][0]) + ',' + str(sess.run(b1)[0]) + ',' + str(sess.run(b2)[0])

def tensorflowAnalysisWithoutContact(data, trees, howFarBack):
    steps = 10000
    learn_rate = 0.03
    dataFrame = createMainDataFrameWithoutContact(data, trees, howFarBack)
    dataFrame = dataFrame.dropna()
    target = np.array(dataFrame['TipState'])
    target = [[member] for member in target]
    x1 = np.array(dataFrame['AncestorState'])
#     x2 = np.array(dataFrame['NeighbourValue'])
    X = np.transpose(np.array([x1]))
    dependent_variables = np.array(X)	
    x = tf.placeholder(tf.float32, [None, 1], name = "x")
    W = tf.Variable(tf.zeros([1, 1]), name = "W")
    b1 = tf.Variable(tf.ones([1]), name = "b1")
    b2 = tf.Variable(tf.ones([1]), name = "b2")
    b = b1 * b2
#     b = tf.Variable(tf.zeros([1]), name = "b")
    y = tf.matmul(x, W) + b
#     y = tf.matmul(x, W)
    y_ = tf.placeholder(tf.float32, [None, 1])
    cost = tf.reduce_mean(tf.square(y_ - y))
    cost_sum = tf.summary.scalar("cost", cost)
    train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)	
    clip_op = tf.assign(W, tf.clip_by_value(W, 0, np.infty))
    clip_op2 = tf.assign(b1, tf.clip_by_value(b1, 0, np.infty))
    clip_op3 = tf.assign(b2, tf.clip_by_value(b2, 0, np.infty))
    sum = tf.reduce_sum(W) + b1
#     sum = tf.reduce_sum(W)
    normalise1 = tf.assign(W, W / sum)
    normalise2 = tf.assign(b1, b1 / sum)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in xrange(steps):
        feed = {x: X, y_: target}
        sess.run(train_step, feed_dict = feed)
        sess.run(clip_op)
        sess.run(clip_op2)
        sess.run(clip_op3)
#         print sess.run(b1)
#         print sess.run(b2)
#         print sess.run(b)
        
#         if sess.run(sum) > 1:
        sess.run(normalise1)
        sess.run(normalise2)
    coefficients = sess.run(W)
    return str(coefficients[0][0]) + ',' + str(sess.run(b1)[0]) + ',' + str(sess.run(b2)[0])

def simpleAnalysis(data, trees, howFarBack, threshold, limit, excludeRelatives = True):
	dataFrame = createSimpleDataFrame(data, trees, howFarBack, threshold, limit, excludeRelatives = excludeRelatives)
	dataFrame = dataFrame.dropna()
	print dataFrame
	y = np.array(dataFrame['TipState'])
	x1 = np.array(dataFrame['RelativesValue'])
	x2 = np.array(dataFrame['NeighbourValue'])
	X = np.transpose(np.array([x1, x2]))
	model = Lasso(alpha = 0.0001, fit_intercept = True, positive = True)
	model = model.fit(X, y)
	return model.coef_, model.intercept_

def simpleAnalysisTensorflow(data, trees, howFarBack, threshold, limit, excludeRelatives = True):
    dataFrame = createSimpleDataFrame(data, trees, howFarBack, threshold, limit, excludeRelatives = excludeRelatives)
    dataFrame = dataFrame.dropna()
    print dataFrame
    x1 = np.array(dataFrame['RelativesValue'])
    x2 = np.array(dataFrame['NeighbourValue'])
    X = np.transpose(np.array([x1, x2]))
    steps = 10000
    learn_rate = 0.03
    target = np.array(dataFrame['TipState'])
    target = [[member] for member in target]
    dependent_variables = np.array(X)	
    x = tf.placeholder(tf.float32, [None, 2], name = "x")
    W = tf.Variable(tf.zeros([2, 1]), name = "W")
#     b = tf.Variable(tf.ones([1]), name = "b")
    b1 = tf.Variable(tf.ones([1]), name = "b1")
    b2 = tf.Variable(tf.ones([1]), name = "b2")
    b = b1 * b2
    y = tf.matmul(x, W) + b
    y_ = tf.placeholder(tf.float32, [None, 1])
    cost = tf.reduce_mean(tf.square(y_ - y))
    cost_sum = tf.summary.scalar("cost", cost)
    train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)	
    clip_op = tf.assign(W, tf.clip_by_value(W, 0, np.infty))
#     clip_op2 = tf.assign(b, tf.clip_by_value(b, 0, np.infty))
    clip_op2 = tf.assign(b1, tf.clip_by_value(b1, 0, np.infty))
    clip_op3 = tf.assign(b2, tf.clip_by_value(b2, 0, np.infty))
    sum = tf.reduce_sum(W) + b1
    normalise1 = tf.assign(W, W / sum)
    normalise2 = tf.assign(b1, b1 / sum)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in xrange(steps):
        feed = {x: X, y_: target}
        sess.run(train_step, feed_dict = feed)
        sess.run(clip_op)
        sess.run(clip_op2)
        sess.run(clip_op3)
        sess.run(normalise1)
        sess.run(normalise2)
    coefficients = sess.run(W)
    return str(coefficients[0][0]) + ',' + str(coefficients[1][0]) + ',' + str(sess.run(b1)[0]) + ',' + str(sess.run(b2)[0])

def simplePCAAnalysisTensorflow(data, trees, componentNumber, howFarBack, threshold, limit, excludeRelatives = True):
    try:
    	dataFrame = pandas.read_csv(componentNumber + 'Dataframe.txt')
    except:
		dataFrame = createSimplePCADataFrame(data, trees, componentNumber, howFarBack, threshold, limit, excludeRelatives = excludeRelatives)
		dataFrame.to_csv(componentNumber + 'Dataframe.txt')
    print dataFrame['TipState']
    print dataFrame['NeighbourValue']
    print dataFrame['RelativesValue']
    dataFrame = dataFrame.dropna()
    x1 = np.array(dataFrame['RelativesValue'])
    x2 = np.array(dataFrame['NeighbourValue'])
    X = np.transpose(np.array([x1, x2]))
    steps = 10000
    learn_rate = 0.03
    target = np.array(dataFrame['TipState'])
    target = [[member] for member in target]
    dependent_variables = np.array(X)	
    x = tf.placeholder(tf.float32, [None, 2], name = "x")
    W = tf.Variable(tf.zeros([2, 1]), name = "W")
#     b = tf.Variable(tf.ones([1]), name = "b")
    b1 = tf.Variable(tf.ones([1]), name = "b1")
    b2 = tf.Variable(tf.ones([1]), name = "b2")
    b = b1 * b2
    y = tf.matmul(x, W) + b
    y_ = tf.placeholder(tf.float32, [None, 1])
    cost = tf.reduce_mean(tf.square(y_ - y))
    cost_sum = tf.summary.scalar("cost", cost)
    train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)	
    clip_op = tf.assign(W, tf.clip_by_value(W, 0, np.infty))
#     clip_op2 = tf.assign(b, tf.clip_by_value(b, 0, np.infty))
    clip_op2 = tf.assign(b1, tf.clip_by_value(b1, 0, np.infty))
    clip_op3 = tf.assign(b2, tf.clip_by_value(b2, 0, np.infty))
    sum = tf.reduce_sum(W) + b1
    normalise1 = tf.assign(W, W / sum)
    normalise2 = tf.assign(b1, b1 / sum)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in xrange(steps):
        feed = {x: X, y_: target}
        sess.run(train_step, feed_dict = feed)
        sess.run(clip_op)
        sess.run(clip_op2)
        sess.run(clip_op3)
        sess.run(normalise1)
        sess.run(normalise2)
    coefficients = sess.run(W)
    return str(coefficients[0][0]) + ',' + str(coefficients[1][0]) + ',' + str(sess.run(b1)[0]) + ',' + str(sess.run(b2)[0])
	
def PCAAnalyseMeanSquaredErrors(componentNumber):
	dataFrame = pandas.read_csv(componentNumber + 'Dataframe.txt')
	dataFrame = dataFrame.set_index('tip')
	dataFrame = dataFrame.dropna()
	dataFrame['RelativesSquaredError'] = (dataFrame['TipState'] - dataFrame['RelativesValue']) ** 2
	dataFrame['NeighboursSquaredError'] = (dataFrame['TipState'] - dataFrame['NeighbourValue']) ** 2
	x = dataFrame['RelativesSquaredError'] - dataFrame['NeighboursSquaredError']
	print x.sort_values(ascending = True)