from sklearn.neural_network import MLPClassifier
from sklearn import tree
from nltk.chunk import *
from sklearn import preprocessing
from sklearn import svm,cross_validation,metrics
import nltk
import numpy as np
import re


label = []
w_word = []
first_word = []
secd_word = []
lengthof_NNP = []
question = []

dic_W = {'Who':1,"When":2,"Where":3,"How":4,"What":5,"Which":6,"Whose":7,"Other":8}
dic_first = {'VBD':1,'NN':2,'VBZ':3,'MD':4,'VBP':5,'JJ':6,'CD':7,'VB':8,'VBG':9,'NNS':10,'PRP':11,'WDT':12,'NNP':13,'RB':14}
dic_lab = {'ABBR':0,'DESC':1,'ENTY':2,'HUM':3,'LOC':4,'NUM':5,'Other':6}
dic_sec = {'PRP$':1,'VBG':2,'VBD':3,'VBP':4,'WDT':5,'JJ':6,'VBZ':7,'DT':8,'NN':9,'POS':10,'.':11,'PRP':12,'RB':13,':':14,'NNS':15,'NNP':16,'VB':17,'CC':18,'VBN':19,'IN':20,'CD':21,'MD':22,'NNPS':23,'JJS':24,'JJR':25}
train_lists = []

def QuestionTest(question):
	Q = question
	token = nltk.word_tokenize(Q)
	pos = nltk.pos_tag(token)
	sets = []
	
	if dic_W.has_key(pos[0][0]):
		sets.append(dic_W[pos[0][0]])
	else:
		sets.append(8)

	
	count = 0
	for item in pos:
		if item[1] == 'NNP':
			count = count + 1
	sets.append(count)

	if dic_first.has_key(pos[1][1]):
		sets.append(dic_first[pos[1][1]])
	else:
		sets.append(15)


	if dic_sec.has_key(pos[2][1]):
		sets.append(dic_sec[pos[2][1]])
	else:
		sets.append(26)

	EXP = np.asarray(sets)
	X = EXP

	result = Class.predict(X)
	
	for key,val in dic_lab.iteritems():
		if val == result:
			return key
	return 'No_question_type'

path = 'Training.txt'
file = open(path)
enc = preprocessing.OneHotEncoder(categorical_features = [0,2,3])



for row in file:
	token = nltk.word_tokenize(row.split(' ',1)[1].replace('\n',''))
	pos = nltk.pos_tag(token)
	label = row.split(':',1)[0]
	sets = []

	if dic_W.has_key(pos[0][0]):
		sets.append(dic_W[pos[0][0]])
	else:
		sets.append(8)

	
	count = 0
	for item in pos:
		if item[1] == 'NNP':
			count = count + 1
	sets.append(count)

	
	if dic_first.has_key(pos[1][1]):
		sets.append(dic_first[pos[1][1]])
	else:
		sets.append(15)

	
	if dic_sec.has_key(pos[2][1]):
		sets.append(dic_sec[pos[2][1]])
	else:
		sets.append(26)

	
	if dic_lab.has_key(label):
		sets.append(dic_lab[label])
	else:
		sets.append(6)
	train_lists.append(sets)

train_lists = np.asarray(train_lists)


Class = tree.DecisionTreeClassifier()
X = train_lists[:,0:4]
y = train_lists[:,4]
Class.fit(X,y)
result = Class.predict(X)
