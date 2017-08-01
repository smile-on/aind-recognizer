import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths # 
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_model = None 
        best_score = 1e6
        for num_components in range(self.min_n_components, self.max_n_components+1):
            model = self.base_model(num_components)
            if model:
                try:
                    logL = model.score(self.X, self.lengths)
                    # total data points in all training samples used by this model
                    num_data_points = len(self.X)
                    # number of free parameters
                    n = model.n_components
                    transition_probs = n * (n - 1) # fully connected HMM 
                    emission_probs = model.means_.size + np.count_nonzero(model.covars_) #  means + covars 
                    num_parameters = transition_probs + emission_probs 
                    # BIC = -2 * logL + p * log(N)
                    score = -2 * logL + num_parameters * math.log(num_data_points)
                    if self.verbose:
                        print(f'bic {score} at {num_components}')
                    if score < best_score: # BIC optimality at minimum score
                        best_score = score
                        best_model = model
                except ValueError: # rows of transmat_ must sum to 1.0 (got [ 1.  1.  1.  0.  1.])
                    break
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    DIC = log(P(X(i)) - 1/(M-1) SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_model = None
        best_score = -1e6
        for num_components in range(self.min_n_components, self.max_n_components+1):
            model = self.base_model(num_components) # train a classifier for self.this_word 
            if model:
                try:
                    # positive score on trained category = log(P(X(i))
                    logL_pos = model.score(self.X, self.lengths) # self.this_word
                    # negative score on incorrect classification = 1/(M-1) SUM(log(P(X(all but i))
                    other_words = [w for w in self.words.keys() if w != self.this_word]
                    neg_scores = []
                    for w in other_words:
                        X, lengths = self.hwords[w]
                        neg_scores.append(model.score(X, lengths))
                    logL_neg = np.mean(neg_scores)
                    # DIC = log(P(X(i)) - 1/(M-1) SUM(log(P(X(all but i))
                    score = logL_pos - logL_neg 
                    if self.verbose:
                        print(f'dic {score} at {num_components} with pos {logL_pos} neg {logL_neg}')
                    if score > best_score: # DIC optimality at maximum score
                        best_score = score
                        best_model = model
                except ValueError: # rows of transmat_ must sum to 1.0 (got [ 1.  1.  1.  0.  1.])
                    break 
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        raise NotImplementedError
