import warnings
from asl_data import SinglesData

from random import random

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for word_id in range(len(test_set.wordlist)): # udacity insists on using id to make guess ;)
        x, lengths = test_set.get_item_Xlengths(word_id)
        # makeing guess
        best_score = float("-Inf")
        best_guess = None
        word_probabilities = dict() # {word: logL}
        for word, model in models.items():
            if model:
                try:
                    logL = model.score(x, lengths)
                except ValueError: 
                    # hmm.n_components > data points in test
                    logL = float("-Inf")
            else:
                print(f'no model for {word}')
                logL = float("-Inf")
            word_probabilities[word] = logL
            if logL > best_score:
                best_score = logL
                best_guess = word
        probabilities.append(word_probabilities)
        guesses.append(best_guess)
    # debug sanity chesck
    # print(f'words {test_set.wordlist[0:10]} \n best {guesses[0:10]}')
    return probabilities, guesses
