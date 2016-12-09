from nltk.corpus import stopwords
import nltk
import operator
import re
import math as m
from fractions import Fraction
from nltk.stem.porter import PorterStemmer


def DocumentRetrieve(path,keywords):
	file = open(path)
	flag = False
	for row in file:

		row = row.replace('\n','')
		row = row.replace('\r','')
		row = row.split()
		for word in row:
			if keywords.has_key(word):
				flag = True
				break
	if flag:
		return path
	else:
		return ''


def TF_IDF(keywords,Relevant_Documents,total_Counts):
	qtf = keywords
	df = Doc_Freq(keywords,Relevant_Documents)
	tf_lists = {}
	for document in Relevant_Documents:
		tf = Term_Freq(keywords,document)
		if tf != {}:
			tf_lists[document] = tf
	TF_IDF = {}
	for key,val in tf_lists.iteritems():
		score = 0
		term = val
		max_term = MaxTerm(key)
		for eachterm,value in term.iteritems():
			TF = 0.5 + (0.5 * value/max_term)
			IDF = m.log(Fraction(total_Counts,df[eachterm]))

			# IDF = m.log(float(total_Counts/df[eachterm]))
			tf_idf = TF * IDF
			score = score + tf_idf
		if score != 0:
			TF_IDF[key] = score
	return 	sorted(TF_IDF.iteritems(), key=operator.itemgetter(1),reverse = True)


def Term_Select(question):
	stop = set(stopwords.words('english'))
	regex = r'[^A-Za-z0-9\s]+'
	question = re.sub(regex,'', question)
	tokenwords = nltk.word_tokenize(question)
	lists = []
	for word in tokenwords:
		if word not in stop:
			lists.append(word)
	pos_tags = nltk.pos_tag(lists)
	Query_Terms = {}
	for item in pos_tags:
		if item[0] != 'What' and item[0] != 'Which' and item[0] != 'When' and item[0] != 'Where' and item[0] != 'Whose' and item[0] != 'Who' and item[0] != 'How' and not Query_Terms.has_key(item[0]):
			Query_Terms[item[0]] = 1
		elif item[0] != 'What' and item[0] != 'Which' and item[0] != 'When' and item[0] != 'Where' and item[0] != 'Whose' and item[0] != 'Who' and item[0] != 'How' and Query_Terms.has_key(item[0]):
			Query_Terms[item[0]] = Query_Terms[item[0]] + 1
		
	return Query_Terms






def MaxTerm(Document):
	file = open(Document)
	MaximumTerms = {}
	stop = set(stopwords.words('English'))
	for row in file:
		row = row.replace('\n','')
		words = row.split(' ')
		for word in words:
			if word not in stop and word != 'I' and word != 'The' and word != 'would' and word != 'Would':
				if not MaximumTerms.has_key(word):
					MaximumTerms[word] = 1
				else:
					MaximumTerms[word] = MaximumTerms[word] + 1
	return max(MaximumTerms.iteritems(), key=operator.itemgetter(1))[1]

def Find_Doc(key_words):
 	Relevant_Documents = []
	for year in range(3,5):
		for month in range(1,13):
			for day in range(1,32):

				if month < 10 and day < 10:
					file_path ='201' + str(year) + '/' + '201' + str(year) + '-0' + str(month) + '-0' + str(day) + '.txt'
				elif month < 10 and day >= 10:
					file_path ='201' + str(year) + '/' + '201' + str(year) + '-0' + str(month) + '-' + str(day) + '.txt'
				elif month > 10 and day < 10:
					file_path ='201' + str(year) + '/' + '201' + str(year) + '-' + str(month) + '-0' + str(day) + '.txt'
				else:
					file_path ='201' + str(year) + '/' + '201' + str(year) + '-' + str(month) + '-' + str(day) + '.txt'
				document = ''
				try:
					document = DocumentRetrieve(file_path,key_words)
				except Exception,e:
					continue
				if document != '':
					Relevant_Documents.append(document)
	return Relevant_Documents





	return tf

def Doc_Freq(queryterms,Relevant_Documents):
	df = dict.fromkeys(queryterms.keys(),0)
	for term in df:
		for document in Relevant_Documents:
			flag = False
			file = open(document)
			for row in file:
				row = row.replace('\n','')
				words = row.split(' ')
				for word in words:
					if word == term:
						df[term] = df[term] + 1
						flag = True
						break
				if flag == True:
					break
	return df

def Term_Freq(queryterms,Document):
	tf = {}
	file = open(Document)
	for row in file:
		row = row.replace('\n','')
		words = row.split(' ')
		for word in words:
			if queryterms.has_key(word) and not tf.has_key(word):
				tf[word] = 1
			elif queryterms.has_key(word) and tf.has_key(word):
				tf[word] = tf[word] + 1

def Doc_Sub(Document,Num):
	subsets = []
	for i in range(0,Num):
		subsets.append(Document[i][0])
	return subsets

question = 'What is the unemployment rate?'
queryterms = Term_Select(question)
print queryterms
Relevant_Documents = Find_Doc(queryterms)
ans = TF_IDF(queryterms,Relevant_Documents,len(Relevant_Documents))

import Generate_Query as Qg
import classifier as QC
import nltk
import operator
import math as m
import re 
from nltk.tokenize import sent_tokenize
from nltk.chunk import *
from nltk.corpus import stopwords
from fractions import Fraction


Document_set = Qg.Doc_Sub(Qg.ans,20)
stop = set(stopwords.words('english'))

def Rule_Based_Answer(Question):
	Qclass = QC.test_questions(Question)
	# rule = {'PERSON':'HUM','GPE':'LOC','NNP':['ENTY','DESC'],'NN':['ENTY','DESC'],'CD':'NUM','NNS':['ENTY','DESC']}
	rule = {'HUM':'PERSON','LOC':'GPE','ENTY':['NNP','NN','NNS'],'DESC':['NNP','NN','NNS'],'NUM':'CD','ABBR':['NNP','NN','NNS']}
	Aclass = rule[Qclass]
	return Aclass

def filterAnsByLabel(Document_set,Aclass):
	A_Backup = []
	for Document in Document_set:
		file = open(Document)
		for row in file:
			row = row.replace('\r','')
			row = row.replace('\n','')
			try:
				sentence = sent_tokenize(row)
			except:
				continue
			for singleSentence in sentence:
				flag = False
				words = singleSentence.split()
				for word in words:
					if Query_term.has_key(word) and flag == False:
						A_Backup.append(singleSentence)
						flag = True

	Fans = []
	for sentence in A_Backup:
		flag = False
		token = nltk.word_tokenize(sentence)
		tags = nltk.pos_tag(token)
		result_ = nltk.ne_chunk(tags,binary = False)
		for item in result_:
			try:
				if item.label() == Aclass and flag == False:
					Fans.append(sentence)
					flag = True
			except:
				if flag == False:
					if len(Aclass) == 3:
						inerflag = False
						for tag in tags:
							if tag[1] == 'CD':
								inerflag = True
						if inerflag == False:
							Fans.append(sentence)
							flag = True
					elif Aclass == 'CD':
						if item[1] == 'CD':
							Fans.append(sentence)
							flag = True
	return Fans

def bigrams(sentence):
	stop = set(stopwords.words('english'))
	Q = nltk.word_tokenize(sentence)
	result = []
	result.extend([i for i in Q if i not in stop])
	result_bigram = list(nltk.bigrams(result))
	return result_bigram

def Answer_Score(Ans_set,question_):
	question = nltk.word_tokenize(question_)
	regex = r'[^A-Za-z0-9\s]+'
	question = re.sub(regex,'',question_)
	Qbigram = list(nltk.bigrams(question))
	Count_Same_Word = 0 
	total = {}
	for sentence in Ans_set:
		label = sentence
		score = 0
		sentence = nltk.word_tokenize(sentence)
		sentence_bigram = list(nltk.bigrams(sentence))
		#score of bigram match
		for bigram in Qbigram:
			for item in sentence_bigram:
				if item == bigram:
					score = score + 2
		#score of unigram match
		for item in question:
			for word in sentence:
				if word == item:
					score = score + 1
		score = Fraction(score,len(sentence))
		total[label] = score
	answer = sorted(total.iteritems(), key = operator.itemgetter(1),reverse = True)
	
	return answer

def TF_IDF(Ans_set,question_):
	NUM = len(Qg.Relevant_Document)
	Qterm = Q.Question_term(question_)
	df = Doc_Freq(Qterm,Qg.Relevant_Document)
	mark = []
	for key,val in df.iteritems():
		if val == 0:
			mark.append(key)
	TFIDF = {}
	print df
	for each_sentence in Ans_set:
		each_sentence = each_sentence.replace(',','')
		each_sentence = each_sentence.replace('|','')

		try:
			sentence_ = nltk.word_tokenize(each_sentence)
		except:
			continue
		sentence = []
		for word in sentence_:
			if word not in stop:
				sentence.append(word)
		
		tf = Term_F(Qterm,sentence)
		mean = Average_Length(Ans_set)
		Num = len(sentence)
		score = 0
		for eachterm,termval in tf.iteritems():
			if termval != 0:
				TF = 0.5 + 0.5 * termval/Num
				IDF = m.log(Fraction(NUM,df[eachterm]))
				# score = score + TF * IDF
				score = score + TF * IDF
			else :
				TF = 0.5 + 0.5 * termval/Num
				IDF = m.log(Fraction(NUM,df[eachterm]))
				# score = score - TF * IDF
				score = score - TF * IDF
		TFIDF[each_sentence] = score
	return sorted(TFIDF.iteritems(), key=operator.itemgetter(1),reverse = True)
	# return TFIDF

# compute document_frequency
def Doc_Freq(queryterms,All_relevant_Document):
	df = dict.fromkeys(queryterms.keys(),0)
	for term in df:
		for document in All_relevant_Document:
			flag = False
			file = open(document)
			for row in file:
				row = row.replace('\n','')
				words = row.split(' ')
				for word in words:
					if word == term:
						df[term] = df[term] + 1
						flag = True
						break
				if flag == True:
					break
	return df
def question_Term(question):
	question = question.replace('?','')
	question_ = nltk.word_tokenize(question)
	tags = nltk.pos_tag(question_)
	question = []
	for item in tags:
		if item[0] != 'Which':
			if item[1] == 'NNS' or item[1] == 'NNP' or item[1] == 'CD' or item[1] == 'NN' or item[1] == 'JJ':
				question.append(item[0])
	Qterm = {}
	
	for item in question:
		if not Qterm.has_key(item):
			Qterm[item] = 1
		else:
			Qterm[item] = Qterm[item] + 1
	return Qterm 

def Term_F(queryterm,sentence):
	tf = dict.fromkeys(queryterm.keys(),0)
	for word in sentence:
		if queryterm.has_key(word):
			tf[word] = tf[word] + 1
	return tf		

def Maximum_Term(sentence):
	Maximum_Term = {}
	for word in sentence:
		if Maximum_Term.has_key(word):
			Maximum_Term[word] = Maximum_Term[word] + 1
		else:
			Maximum_Term[word] = 1
	return max(Maximum_Term.iteritems(), key = operator.itemgetter(1))[1]

def Doc_sub(subset,Num):
	subset = []
	for i in range(0,Num):
		result.append(subset[i][0])
	return subset

def Average_Length(Allset):
	Idx = 0
	length = 0
	regex = r'[^A-Za-z0-9\s]+'
	for sentence in Allset:
		sentence = re.sub(regex,'',sentence)
		result = []
		for word in sentence:
			if word not in stop:
				result.append(word)
		length = length + len(result)
		Idx = Idx + 1
	return Fraction(length,Idx)


Query_term = dict.fromkeys(Qg.queryterms.keys(),0)
Ans_type = Rule_Based_Answer(Qg.question)

ansset = filterAnsByLabel(Document_set,Ans_type)

ans = TF_IDF(ansset,Qg.question)
ans_sub = Doc_sub(ans,10)
print ans_sub

import re
import nltk
import Query_generation as Qg
import AnswerGeneration as AG

question = Qg.question
token = nltk.word_tokenize(question)
pos_tags = nltk.pos_tag(token)
CEO = 'ceo.csv'
Companies = 'companies.csv'
ans = AG.ans_sub

file_ceo = open(CEO)
file_com = open(Companies)

CEO_dic = {}
COM_dic = {}

for row in file_ceo:
	row = row.replace(',',' ')
	dic = row.split('\r')
	for item in dic:
		if not CEO_dic.has_key(item):
			CEO_dic[item] = 1

for row in file_com:
	row = row.replace(',',' ')
	dic = row.split('\r')
	for item in dic:
		if not COM_dic.has_key(item):
			COM_dic[item] = 1

flag = ''
for item in pos_tags:
	if item[0] == 'CEO':
		flag = 'ceo'
		break
	elif item[0] == 'company':
		flag = 'company'
		break
final_ans = ''

if flag == 'ceo':
	for sentence in ans:
		token = nltk.word_tokenize(sentence)
		result_bigram = list(nltk.bigrams(token))
		for bigram in result_bigram:
			name = str(bigram[0]) + ' ' + str(bigram[1])
			if CEO_dic.has_key(name):
				final_ans = name

elif flag == 'company':
	final_ans = []
	for sentence in ans:
		pattern = r'[^A-Za-z0-9\s]+'
		sentence = re.sub(pattern,'', sentence) 
		token = nltk.word_tokenize(sentence)
		result_bigram = list(nltk.bigrams(token))
		result_trigram = list(nltk.ngrams(token,3))
		result_fourgram = list(nltk.ngrams(token,4))
		for item in token:
			if COM_dic.has_key(item):
				final_ans = item
		for bigram in result_bigram:
			name = str(bigram[0]) + ' ' + str(bigram[1])
			if COM_dic.has_key(name):
				final_ans = item
		for bigram in result_trigram:
			name = str(bigram[0]) + ' ' + str(bigram[1]) + ' ' + str(bigram[2])
			if COM_dic.has_key(name):
				final_ans = item
		for bigram in result_fourgram:
			name = str(bigram[0]) + ' ' + str(bigram[1]) + ' ' + str(bigram[2]) + ' ' + str(bigram[3])
			if COM_dic.has_key(name):
				final_ans = item

print question
print final_ans

