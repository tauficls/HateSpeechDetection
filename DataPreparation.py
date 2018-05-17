import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import collections

class PreparingData:

	def read2data(self, fileHate, fileNoHate, header):
		# print(self.fileHate)
		dataHate = pd.read_csv(fileHate, header=header)
		# print(self.fileNoHate)
		dataNoHate = pd.read_csv(fileNoHate, header=header)
		data = pd.concat([dataHate, dataNoHate])
		data = data.sample(frac=1).reset_index(drop=True)
		# data = shuffle(data)
		return data

	def train_test_split(self, train, value, test_size):
		return train_test_split(train, value, test_size=test_size)

	def read_data(self, filename, header):
		data = pd.read_csv(filename, header=header)#, n_rows=2000)
		return data


	def split_akustic_lld(dataTrain, dataTest):
		return dataTrain.iloc[:,1:-1], dataTest.iloc[:,1:-1], dataTrain["class"], dataTest["class"]

	def split_akustic_func(dataTrain, dataTest):
		return dataTrain.iloc[:,2:-1], dataTest.iloc[:,2:-1], dataTrain["class"], dataTest["class"]	

	

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import Word2Vec

class WordEmbedding:

	def __init__(self, sentences=None):
		self.model = None
		self.sentences = sentences

	def train_cbow(self):
		model = Word2Vec(self.sentences, min_count=10, sg=0, size=50, workers=4)
		self.model = model
		return model

	def train_skipgram(self):
		model = Word2Vec(self.sentences, min_count=1, sg=1, size=50)
		self.model = model
		return model

	def saveWordEmbedding(self, filename):
		self.model.save(filename)

	def loadWordEmbedding(self, filename):
		return Word2Vec.load(filename)

	def get_model_embedding(self):
		return self.model

class Tokenize:

	def __init__(self, dataset):
		self.dataset = dataset

	def tokenize(self):
		print("Tokenize")
		result = []

		for index, data in self.dataset.iterrows():
			tokenized_words = word_tokenize(data["Transkrip"].lower())
			# tokenized_words = word_tokenize(data[0].lower())
			result.append(tokenized_words)

		self.dataset["tokenize"] = result

		return self.dataset

	def tokenize_rid_punct(self):
		print("Data Preparation")
		
		result = []

		datas = self.dataset.split(" ")
		for n in datas:
			if n.isdigit():
				self.dataset = self.dataset.replace(n, "1")
		# tokenized_words = [ i for i in RegexpTokenizer(r'\w+').tokenize(data.lower()) if i not in stop]
		tokenized_words = RegexpTokenizer(r'\w+').tokenize(self.dataset.lower())
		result.append(" ".join(tokenized_words))
		# result.append(tokenized_words)
		self.dataset = result

		return self.dataset

# class Preprocessing:

# 	def Tokenize(self, dataset, columnName):
# 		print("Tokenize")
# 		result = []
# 		# for word in stopwords.words('indonesian'):
# 		# 	print(word)

# 		for index, data in df.iterrows():
# 			tokenized_words = word_tokenize(data[columnName].lower())
# 			# tokenized_words = [word for word in tokenized_words if not word in stopwords.words('indonesian') and not word[0].isdigit()]
# 			result.append(tokenized_words)

# 		df["transkrip_tokenize"] = result

# 		return df

	# def Embedding(self, sentences, type):
	# 	wordEmbedding = WordEmbedding()

	# 	if (type == 1):
	# 		# CBOW
	# 		print("Word Embedding CBOW")
	# 		model = wordEmbedding.trainWord2Vec(sentences, 0)
	# 		wordEmbedding.saveWordEmbedding("model_CBOW.bin")
	# 	elif (type == 2):
	# 		# Skip gram
	# 		print("Word Embedding Skip Gram")
	# 		model = wordEmbedding.trainWord2Vec(sentences, 1)
	# 		wordEmbedding.saveWordEmbedding("model_SkipGram.bin")

	# 	return model