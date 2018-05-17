from DataPreparation import PreparingData
from DataPreparation import WordEmbedding
from DataPreparation import Tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from keras.preprocessing import sequence
from sklearn import preprocessing
from keras.models import model_from_json
import numpy as np
import pandas as pd
import pickle
import os
from keras import backend as K
from sklearn.metrics import accuracy_score, classification_report
import sys

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
script_dir = os.path.dirname(__file__)
# data_dir = '/home/prosa/data/speech/hatespeech/'
# script_dir = '/home/prosa/src/hatespeech/'

def evaluate(y_test, y_pred):
	print("Accuracy : " + str(accuracy_score(y_test, y_pred)))
	print("Classification Report : ")
	print(classification_report(y_test, y_pred, digits=4))

def vectorizer(X_train, X_test, n_gram=(1,1)):
	if (mode == 'count_vectorizer'):
		vectorizer = CountVectorizer(ngram_range=(1,1), tokenizer=word_tokenize, lowercase=True)
	elif (mode == 'tfidf_vectorizer'):
		vectorizer = TfidfVectorizer(ngram_range=n_gram, tokenizer=word_tokenize, lowercase=True)
	X_train = vectorizer.fit_transform(X_train)
	X_test = vectorizer.transform(X_test)

	return X_train, X_test

def meanEmbedding(embedding_model, X, em_size):
	oov = np.array([])
	total_word = 0
	temp = np.array([])
	for data in X:
		list_word = np.array([])
		mean = np.array([])
		words = data.split(" ")
		total_word = total_word + len(words)
		for word in words:
			if word in embedding_model.wv.vocab:
				list_word = np.append(list_word, embedding_model[word])
			else:
				oov = np.append(oov, word)
				list_word = np.append(list_word, np.zeros(em_size))

		length = list_word.shape[0]
		list_word = list_word.reshape(length//em_size, em_size)

		for x in list_word.transpose():
			mean = np.append(mean, np.mean(x))

		temp = np.append(temp, mean)

	temp = temp.reshape(temp.shape[0]//em_size, em_size)

	oov = np.unique(oov)
	total_oov = oov.shape[0]

	print("Total oov : ", str(total_oov))
	print("Total words : ", str(total_word))

	print("percentage oov : ", str(total_oov/total_word))
	print("Words oov :")
	print(oov)

	return temp

def embedding(embedding_model, X, em_size, pad_size):
	oov = np.array([])
	# total = np.array([])
	total_word = 0
	temp = np.array([])
	for data in X:
		list_word = np.array([])
		words = data.split(" ")
		total_word = total_word + len(words)
		# total = np.append(total, len(words))
		for word in words:
			if word in embedding_model.wv.vocab:
				
				list_word = np.append(list_word, embedding_model[word])
			else:
				
				oov = np.append(oov, word)
				list_word = np.append(list_word, np.zeros(em_size))

		length = list_word.shape[0]
		list_word = list_word.reshape(length//em_size, em_size)
		list_word = sequence.pad_sequences(list_word.transpose(), maxlen=pad_size)
		if (temp.size):
			temp = np.vstack((temp, [list_word.transpose()]))
		else:
			temp = np.array([list_word.transpose()])	

	oov = np.unique(oov)
	total_oov = oov.shape[0]

	print("Total oov : ", str(total_oov))
	print("Total words : ", str(total_word))
	# print(len(X))
	# print("Mean long text : ", str(total_word/len(X)))
	# print("max words : ", str(np.amax(total)))
	# print("min words : ", str(np.amin(total)))
	print("percentage oov : ", str(total_oov/total_word))
	print("Words oov :")
	print(oov)

	return temp
def read_one_data(file):
	preparing = PreparingData()

	data = preparing.read_data(file, 0)

	return data
def read_data(file, filetest):
	preparing = PreparingData()
	# dataTrain = preparing.read_data(fileHate, fileNoHate, 0)
	# dataTest = preparing.read_data(fileHateTest, fileNoHateTest, 0)
	
	dataTrain = preparing.read_data(file, 0)
	dataTest = preparing.read_data(filetest, 0)

	# print(dataTrain.head())
	# print(dataTest.head())

	return dataTrain, dataTest

def read_dataset():
	file = data_dir + "dataTrain.csv"
	filetest = data_dir + "dataTest.csv"
	return read_data(file, filetest)

def read_prosody_func():
	file = data_dir + "prosody/prosodyAcf_func.csv"
	filetest = data_dir + "prosody/prosodyAcf_func_test.csv"
	return read_data(file, filetest)

def read_mfcc0_func():
	file = data_dir + "MFCC/MFCC12_0_D_A_func.csv"
	filetest = data_dir + "MFCC/MFCC12_0_D_A_func_test.csv"
	return read_data(file, filetest)

def read_mfcce_func():
	file = data_dir + "MFCC/MFCC12_E_D_A_func.csv"
	filetest = data_dir + "MFCC/MFCC12_E_D_A_func_test.csv"
	return read_data(file, filetest)

def read_is09_func():
	file = data_dir + "Interspeech/IS09_emotion_func.csv"
	filetest = data_dir + "Interspeech/IS09_emotion_func_test.csv"
	return read_data(file, filetest)

def read_is10_func():
	file = data_dir + "Interspeech/IS10_paraling_func.csv"
	filetest = data_dir + "Interspeech/IS10_paraling_func_test.csv"
	return read_data(file, filetest)

def read_prosody_lld():
	file = data_dir + "prosody/prosodyAcf_lld.csv"
	filetest = data_dir + "prosody/prosodyAcf_lld_test.csv"
	return read_data(file, filetest)

def read_mfcc0_lld():
	file = data_dir + "MFCC/MFCC12_0_D_A.csv"
	filetest = data_dir + "MFCC/MFCC12_0_D_A_test.csv"
	return read_data(file, filetest)

def read_mfcce_lld():
	file = data_dir + "MFCC/MFCC12_E_D_A.csv"
	filetest = data_dir + "MFCC/MFCC12_E_D_A_test.csv"
	return read_data(file, filetest)

def read_is09_lld():
	file = data_dir + "Interspeech/IS09_emotion_lld.csv"
	filetest = data_dir + "Interspeech/IS09_emotion_lld_test.csv"
	return read_data(file, filetest)

def read_is10_lld():
	file = script_dir + "static/arff/IS10_paraling_lld.csv"
	filetest = script_dir + "static/arff/IS10_paraling_lld_test.csv"
	return read_data(file, filetest)
	
def read_text_corpus():
	preparing = PreparingData()
	data_dir = os.path.dirname(__file__)
	files = os.path.join(data_dir, "textcorpus/xaa")
	data = preparing.read_data(files, None)
	for file in os.listdir("textcorpus"):
		print(file)
		if ("xaa" != file):
			files = os.path.join(data_dir, "textcorpus/" + file)
			data = pd.concat([data, preparing.read_data(files, None)])
	return data

def preprocess(dataset, pad_size, n_features):
	
	# value = dataTrain["name"][0].split("/")[3].split(".")[0]
	total = np.array([])
	# total = np.append(total, len(words))
	value = dataset["name"][0]
	matrix = np.array([])
	y = np.array([])
	temp = np.array([])
	for index, data in dataset.iterrows():
		# values = data["name"].split("/")[3].split(".")[0]
		values = data["name"]
		if (value == values):
			temp = np.append(temp, list(data.iloc[2:-1]))
		else:
			size = temp.shape[0]//n_features
			total = np.append(total, size)
			temp = temp.reshape(size, n_features)
			
			if (temp.shape[0] < pad_size):
				temp = np.pad(temp, ((pad_size - temp.shape[0], 0), (0, 0)), 'constant')
			elif (temp.shape[0] > pad_size):
				deleted_size = temp.shape[0] - pad_size
				count = 0
				while (count < deleted_size):
					temp = np.delete(temp, 0, 0)
					count += 1
					if (count < deleted_size):
						temp = np.delete(temp, -1, 0)
						count += 1
				
			temp = np.array([temp])
			if (matrix.size):
				matrix = np.vstack((matrix, temp))
			else:
				matrix = temp
					
			kelas = value.split("_")[0].replace("'", "")
			if (kelas == "H"):
				y = np.append(y, 1)
			else:
				y = np.append(y, 0)

			temp = np.array([])
			temp = np.append(temp, list(data.iloc[2:-1]))
			value = values
				
	size = temp.shape[0]//n_features
	total = np.append(total, size)
	
	temp = temp.reshape(size, n_features)
	
	if (temp.shape[0] < pad_size):
		temp = np.pad(temp, ((pad_size - temp.shape[0], 0), (0, 0)), 'constant')
	elif (temp.shape[0] > pad_size):
		deleted_size = temp.shape[0] - pad_size
		count = 0
		while (count < deleted_size):
			temp = np.delete(temp, 0, 0)
			count += 1
			if (count < deleted_size):
				temp = np.delete(temp, -1, 0)
				count += 1
				
	temp = np.array([temp])
	if (matrix.size):
		matrix = np.vstack((matrix, temp))
	else:
		matrix = temp

	kelas = value.split("_")[0].replace("'", "")
	if (kelas == "H"):
		y = np.append(y, 1)
	else:
		y = np.append(y, 0)

	print("Max length : ", str(np.amax(total)))
	print("Min length : ", str(np.amin(total)))
	print("Average length : ", str(np.mean(total)))
		
	return matrix, y

def predict_hatespeech(X_test, model_file=None):
	if (model_file != None):
		saved_model = script_dir + "/static/model/lstm/" + model_file

	json_file = open(saved_model + ".json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(saved_model + ".h5")
	print(loaded_model)

	y_pred = loaded_model.predict_classes(X_test)
	K.clear_session()
	return y_pred
	
def text(X_test_text, cbow_file=None, model_file=None):

	tokenizer_test = Tokenize(X_test_text)
	X_test_text = tokenizer_test.tokenize_rid_punct()
	print(X_test_text)
	y_pred = None
	
	if (cbow_file != None):
		cbow = script_dir + "/static/model/word2vec/" + cbow_file
		word2vec = WordEmbedding()
		word2vec_model = word2vec.loadWordEmbedding(cbow)
		em_vector_size = 200
		word_pad_size = 50
		X_test_text = embedding(word2vec_model, X_test_text, em_vector_size, word_pad_size)
		y_pred = predict_hatespeech(X_test_text, model_file)

	return y_pred

def akustik(filename, model_file=None):
	
	for file in filename:
		if ("lld" in file):
			file_lld = file
	dataTest_akustik = read_one_data(file_lld)
	
	frame_pad_size = 750
	n_features = dataTest_akustik.iloc[:, 2:-1].shape[1]
	print("==================================")
	print("Preprocessing Test")
	print("==================================")
	X_test_akustik, y_test_akustik = preprocess(dataTest_akustik, frame_pad_size, n_features)
	# lstm = lstm_akustik(X_train, y_train, max_length=pad_size, n_features=n_features)
	y_pred = predict_hatespeech(X_test_akustik, model_file)
	return y_pred

# def akustik():

def fuse(X_test_text, filename, model_fuse=None, model_cbow=None):
	tokenizer_test = Tokenize(X_test_text)
	X_test_text = tokenizer_test.tokenize_rid_punct()
	print(X_test_text)

	if (model_cbow != None):
		cbow = script_dir + "/static/model/word2vec/" + model_cbow
		word2vec = WordEmbedding()
		word2vec_model = word2vec.loadWordEmbedding(cbow)
		em_vector_size = 200
		word_pad_size = 40
		X_test_text = embedding(word2vec_model, X_test_text, em_vector_size, word_pad_size)
		

	for file in filename:
		if ("lld" in file):
			file_lld = file
	dataTest_akustik = read_one_data(file_lld)


	frame_pad_size = 750
	n_features = dataTest_akustik.iloc[:, 2:-1].shape[1]
	print("==================================")
	print("Preprocessing Test")
	print("==================================")
	X_test_akustik, y_test_akustik = preprocess(dataTest_akustik, frame_pad_size, n_features)
	y_pred = predict_hatespeech([X_test_text, X_test_akustik] , model_fuse)

	return y_pred
	

def main():
# 	# dataTrain_text, dataTest_text = read_dataset()
	filetest = script_dir + "static/arff/IS10_paraling_lld_test.csv"
	dataTest_akustik = read_one_data(filetest)

	model_file = "IS10_1"
	frame_pad_size = 750
	n_features = dataTest_akustik.iloc[:, 2:-1].shape[1]
	print("==================================")
	print("Preprocessing Test")
	print("==================================")
	X_test_akustik, y_test_akustik = preprocess(dataTest_akustik, frame_pad_size, n_features)
	
	y_pred = predict_hatespeech(X_test_akustik, model_file)
	print(y_pred)
	evaluate(y_test_akustik, y_pred)
	
# 	# # Save/Load model
	
# if __name__ == "__main__":
# 	main()
