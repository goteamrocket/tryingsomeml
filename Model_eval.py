from datetime import datetime
import gensim
from gensim.models import doc2vec
import pickle
from random import shuffle
import random
import logging
import pandas as pd
import numpy as np
import multiprocessing
cores = multiprocessing.cpu_count()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
DEBUG = True
# Paths
texts_path =
model_save_path = 
model = gensim.models.Doc2Vec.load(model_save_path)
class Model(object):
    def __init__(self, name, text_path, dm, size, window,  min_count, sample, epochs,
                 hs, negative, dm_mean, dm_concat, dbow_words, alpha, seed):
        self.name = name
        self.dm = dm  # training algorithm. (`dm=1`), 'distributed memory' (PV-DM) is used.
        # Otherwise, `distributed bag of words` (PV-DBOW) is employed.
        self.size = size  # dimensionality of the feature vectors.
        self.window = window  # maximum distance between the predicted word and context words
        self.min_count = min_count  # ignore all words with total frequency lower than this.
        self.sample = sample
        self.epochs = epochs  # `iter` = number of iterations (epochs) over the corpus.
        self.hs = hs  # if 1 (default), hierarchical sampling will be used for model training
        self.negative = negative  # if > 0, negative sampling will be used, the int for negative
        # specifies how many "noise words" should be drawn (usually between 5-20).
        self.dm_mean = dm_mean  # `dm_mean` = if 0 (default), use the sum of the vectors. If 1, use the mean.
        # Only applies when dm is used in non-concatenative mode.
        self.dm_concat = dm_concat  # if 1, use concatenation of context vectors rather than sum / average;
        # default is 0 (off).
        self.dbow_words = dbow_words
        self.alpha = alpha  # initial learning rate (will linearly drop to zero as training progresses).
        self.seed = seed  # random number generator.
        self.texts_path = text_path
        self.report_path = '/home/aviadmin/chernousov/shared/One/report'
        self.report = None
        self.tokenized_texts = None
        self.model = None
        self.learning_data = None
        self.percentage = None
        self.mistakes = None
    def prepare_data(self):
        # Load file with result of preprocessing
        with open(self.texts_path, 'rb') as fp:
            tokenized_texts = pickle.load(fp)
            self.tokenized_texts = tokenized_texts
            counter = 0
            # Make list of texts and keys
            learning_data = []
            for x in range(len(tokenized_texts)):
                # If text not empty
                if tokenized_texts[x][1]:
                    # text, key
                    learning_data.append([tokenized_texts[x][1], [str(tokenized_texts[x][0])]])
                    learning_data.append([tokenized_texts[x][1], [str(tokenized_texts[x][0]) + '_']])
                    #print(tokenized_texts[x])
                else:
                    #print(tokenized_texts[x])
                    counter += 1
        print('total documents: ', len(learning_data))
        print('Empty documents: ', counter)
        self.learning_data = learning_data
    def train_model(self):
        self.prepare_data()
        sentences = [doc2vec.TaggedDocument(words=words, tags=label) for words, label in self.learning_data]
        self.model = gensim.models.Doc2Vec(documents=sentences,
                                           workers=cores,
                                           dm=self.dm,  # training algorithm. (`dm=1`), 'distributed memory' (PV-DM)
                                           #  is used. Otherwise, `distributed bag of words` (PV-DBOW) is employed.
                                           size=self.size,  # dimensionality of the feature vectors.
                                           window=self.window,  # maximum distance between the predicted
                                           # word and context words
                                           alpha=self.alpha,
                                           # initial learning rate (will linearly drop to zero as training progresses).
                                           seed=self.seed,  # random number generator.
                                           min_count=self.min_count,  # ignore words with  frequency lower than this.
                                           sample=self.sample,  # threshold for configuring higher-frequency words
                                           #  downsampling
                                           # default is 0 (off), useful value is 1e-5.
                                           iter=1,  # `iter` = number of iterations (epochs) over the corpus.
                                           hs=self.hs,  # if 1 (default), hierarchical sampling will be used
                                           negative=self.negative,  # if > 0, negative sampling will be used,
                                           #  the int for negative
                                           # specifies how many "noise words" should be drawn (usually between 5-20).
                                           dm_mean=self.dm_mean,
                                           # `dm_mean` = if 0 (default), use the sum of the vectors. If 1, use the mean.
                                           # Only applies when dm is used in non-concatenative mode.
                                           dm_concat=self.dm_concat,
                                           # if 1, use concatenation of context vectors rather than sum / average;
                                           # default is 0 (off).
                                           dbow_words=self.dbow_words
                                           # if set to 1 trains word - vectors( in skip - gram fashion) simultaneous
                                           # with DBOW training; default is 0 (faster training of doc-vecs only).
                                           )
        # Learning cycle
        for epoch in range(self.epochs):
            shuffle(sentences)
            self.model.train(sentences)
            self.model.alpha -= 0.002  # decrease the learning rate
            self.model.min_alpha = model.alpha  # fix the learning rate, no decay
    def evaluate_model(self):
        """
        Counts how many same docs with different tags matched as the same
        :return:
        """
        keys = [str(each[0]) for each in self.tokenized_texts if str(each[0]).isnumeric()]
        sim_keys = []
        counter = 0
        # Make list of most similar to each original documents
        for key in keys:
            similarity_result = self.model.docvecs.most_similar(key, topn=1)
            similar_text_key = similarity_result[0][0]
            sim_keys.append(similar_text_key)
        for key, sim_key in zip(keys, sim_keys):
            if key != sim_key[:-1]:
                counter += 1
        self.mistakes = counter
        self.percentage = counter / len(keys) * 100
        print('Mistakes:', counter)
        print('Matching percent: ', 100 - counter / len(keys) * 100)
    def save_report(self):
        with open(self.report_path, 'rb+') as fp:
            reports = pickle.load(fp)
            new_report = np.array([self.name, self.percentage, self.dm, self.size, self.window,
                                   self.min_count, self.sample, self.epochs, self.negative, self.dm_mean,
                                   self.dm_concat, self.dbow_words, self.alpha, self.seed]).reshape((1, 14))
            new_report = pd.DataFrame(data=new_report, columns=['name', 'eval', 'dm', 'vector size', 'window',
                                                                'min_count', 'sample', 'epochs', 'negative', 'dm_mean',
                                                                'dm_concat', 'dbow_words', 'alpha', 'seed'])
            frames = [reports, new_report]
            self.report = pd.concat(frames)
        pickle.dump(self.report, open("report", "wb+"))
    def show_report(self):
        print('Showing reports: \n')
        with open(self.report_path, 'rb+') as fp:
            reports = pickle.load(fp)
            print(reports)
    #def get_TSNE(self):
    #def get_PCA(self):
    #def save_model_to_file(self, save_path):
    #def load_from_file(self, load_path):
for x in range(100):
    random.seed()
    name_init = str(datetime.now())
            #name, dm, size, window, min_count, sample, epochs,
            #hs, negative, dm_mean, dm_concat, dbow_word
    one = Model(name_init, texts_path,
                    random.randint(0, 1),  # dm
                    random.randint(40, 1000),  # size
                    random.randint(3, 20),  # window
                    random.randint(0, 2),  # min_count
                    random.randint(0, 1)/10000,  # threshold for configuring higher-frequency words downsampling
                    # default is 0 (off), useful value is 1e-5.
                    random.randint(2, 20),  # epochs
                    random.randint(0, 1),  # hs
                    random.randint(5, 40),  # negative
                    0,  # dm_mean
                    1,  # dm_concat
                    random.randint(0, 1),  # dbow_word
                    0.005,  # alpha
                    50)  # seed
    one.train_model()
    one.evaluate_model()
    one.save_report()
    one.show_report()
