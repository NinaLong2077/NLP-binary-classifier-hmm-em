
# File: hmm.py
# Purpose:  Starter code for building and training an HMM in CSC 246.

import time 
import argparse  
import os
import numpy as np
import matplotlib.pyplot as plt
from distutils.util import strtobool

# A utility class for bundling together relevant parameters - you may modify if you like.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# num_states -- this should be an integer recording the number of hidden states
#
# pi -- this should be the distribution over the first hidden state of a sequence
#
# transitions -- this should be a num_states x num_states matrix of transition probabilities
#
# emissions -- this should be a num_states x vocab_size matrix of emission probabilities
#              (i.e., the probability of generating token X when operating in state K)
#
# vocab_size -- this should be an integer recording the vocabulary size; 255 is a safe upper bound
#
# Note: You may want to add fields for expectations.

# transitions: corresponds to the matrix A
# emissions: corresponds to matrix B, defined as p(xn|zn, φ)

'''
references: 
https://github.com/ducanhnguyen/hidden-markov-model/blob/master/src/hmmd.py
https://web.stanford.edu/~jurafsky/slp3/A.pdf
https://github.com/vrjkmr/hmm/blob/master/hmm.py
'''

class HMM:
    #__slots__ = ('pi', 'transitions', 'emissions', 'num_states', 'vocab_size')
    # The constructor should initalize all the model parameters.
    # you may want to write a helper method to initialize the emission probabilities.

    def __init__(self, num_hidden_states, vocabulary):

        '''
        num_hidden_states:
        vocab: unique vocabulary (unique chars)
        '''
        self.num_hidden_states = num_hidden_states
        self.vocabulary_size = vocabulary.shape[0]
        self.vocabulary = vocabulary
        self.A = []
        self.B = []
        self.pi = []


    def fit(self, dataset, num_iter=10, criterion=0.00001):
        LL_list = []
        old_avg_LL = self.LL(dataset)

        LL_list.append(old_avg_LL)

        print("Before EM, initial log likelihood is {:.4f}".format(old_avg_LL))

        for iter in range(num_iter):

          print("iteration {}".format(iter))
          tic = time.time()
          self.em_step(dataset)
          toc = time.time()
          print("Time Elapsed for this iteraion: ", toc - tic)

          cur_avg_LL = self.LL(dataset)

          LL_list.append(cur_avg_LL)
          print("iteration {}, log likelihood is {:.4f}".format(iter, cur_avg_LL))
          
          #LL_diff = old_avg_LL - cur_avg_LL
          LL_diff =  cur_avg_LL - old_avg_LL

          old_avg_LL = cur_avg_LL

          #if  LL_diff < criterion:
           # print("percent of average log liklihood change is smaller than criterion {}, break".format(criterion))
            #break
        
        return LL_list

    def initialize_A_B_pi(self, initial_prob = 'random'):
        '''
        set:
        A: transition matrix (num_hidden_states * num_hidden_states)
        B: emission matrix (num_hidden_states * vacabulary_size)
        pi: initial state probability (num_hidden_states * 1)
        '''
        if initial_prob == 'random':
          # A[i,j]: probability of moving from state i to state j
          # each row sum == 1
          A = np.random.rand(self.num_hidden_states, self.num_hidden_states)
          A = A / A.sum(axis = 1, keepdims = True)

          # B[i,j]: probability of observation j being generated from state i
          B = np.random.rand(self.num_hidden_states, self.vocabulary_size)
          B = B / B.sum(axis = 1, keepdims = True)

          #pi[i]: probability of the Markov chain will start in state i
          pi = np.random.rand(self.num_hidden_states)
          pi = pi / pi.sum()
        
        elif initial_prob == 'uniform': #not real uniform, otherwise em wont really work
          #print("dafafkaldfja")
          A = np.ones((self.num_hidden_states, self.num_hidden_states)) + np.random.rand(self.num_hidden_states, self.num_hidden_states)*2
          A = A / A.sum(axis = 1, keepdims = True)   

          B = np.ones((self.num_hidden_states, self.vocabulary_size)) + np.random.rand(self.num_hidden_states, self.vocabulary_size)*2
          B = B / B.sum(axis = 1, keepdims = True)
          
          pi = np.ones(self.num_hidden_states)+np.random.rand(self.num_hidden_states)*2
          pi = pi / pi.sum()

        self.A = A
        self.B = B
        self.pi = pi

    def _get_observation_idx(self, observation):
        '''
        observation: a char
        return: a index 
        '''
        # return vocabulary index of the observation
        observation_idx = np.argwhere(self.vocabulary == observation)
        return observation_idx.flatten().item()
    
    def compute_alpha(self,observed_seq): 
        num_observed_seq = len(observed_seq)        
        alpha = np.zeros((self.num_hidden_states, num_observed_seq))
        c = np.zeros(num_observed_seq)

        #initialization step, compute alpha[i, 0]        
        o_0 = self._get_observation_idx(observed_seq[0])     
        alpha[:, 0] = self.pi * self.B[:, o_0]

        c[0] = np.sum(alpha[:, 0]) 
        if c[0] == 0:
          return alpha, c
        c[0] = 1/c[0]
        alpha[:, 0] *= c[0]
        
        #recursion step, compute alpha[i, t]
        for t in range(1, num_observed_seq):
          o_t = self._get_observation_idx(observed_seq[t])

          alpha[:, t] = alpha[:, t-1] @ self.A * self.B[:, o_t]

          # alpha[:, t] += np.sum(alpha[:, t-1] * self.A[:, :], axis=1)
          # alpha[:, t] *= self.B[:, o_t] 
          c[t] += np.sum(alpha[:, t])

          #scale alpha[i, t]
          if c[t] == 0:
            return alpha, c
          c[t] = 1/c[t]
          alpha[:,t] *= c[t]

        return alpha, c

    def compute_beta(self,observed_seq, c):

      num_observed_seq = len(observed_seq)        
      beta = np.zeros((self.num_hidden_states, num_observed_seq))
      
      #initialization, 
      beta[:, -1] = c[-1]
      
      # #beta pass
      for t in range(num_observed_seq-2, -1, -1):
         o_t_plus_1 = self._get_observation_idx(observed_seq[t+1])
         beta[:,t] = self.A @ (self.B[:, o_t_plus_1] * beta[:, t+1])
        #  beta[:, t] += np.sum(self.A * self.B[:, o_t_plus_1] * beta[:, t+1], axis=1)
         beta[:, t] *= c[t]
    
      return beta, c
    
    # return the avg loglikelihood for a complete dataset (train OR test) (list of arrays)
    def LL(self, dataset):
        avg_LL = 0
        for observed_seq in dataset:         
          avg_LL += self.LL_helper(observed_seq)
         
        return avg_LL/len(dataset)

    # return the LL for a single sequence (numpy array)
    def LL_helper(self, observed_seq):
        _, c = self.compute_alpha(observed_seq)
        if not np.all(c):
          return 0
        return -np.sum(np.log(c))

    # apply a single step of the em algorithm to the model on all the training data,
    # which is most likely a python list of numpy matrices (one per sample).
    def em_step(self, dataset):
        example_count = 0
        
        accu_A = np.zeros((self.num_hidden_states, self.num_hidden_states))
        accu_B = np.zeros((self.num_hidden_states, self.vocabulary_size))
        accu_pi = np.zeros((self.num_hidden_states))
      
        for observed_seq in dataset:  
          example_count += 1
          if (example_count % 1000 == 0):
            print("sample ", example_count)
         
          local_A, local_B, local_pi = self.em_step_helper(observed_seq)
     
          accu_A += local_A
          accu_B += local_B
          accu_pi += local_pi

        self.A = accu_A/example_count
        self.B = accu_B/example_count
        self.pi = accu_pi/example_count
            
    #apply single step of the em on one training data
    def em_step_helper(self, observed_seq): 
        
        #E step 
        alpha, c = self.compute_alpha(observed_seq)
        beta, _ = self.compute_beta(observed_seq, c)
        gamma, digamma = self.compute_gamma_digamma(alpha, beta, observed_seq,c)

        #M step  
        newA = self.re_estimate_A(gamma, digamma, observed_seq)
        newB = self.re_estimate_B(gamma, observed_seq)
        new_pi = self.re_estimate_pi(gamma)

        return newA, newB, new_pi

        
    def compute_gamma_digamma(self, alpha, beta, observed_seq,c):
        '''
        gamma[j,t] = probability of being in state j at time t
        '''
        num_observed_seq = len(observed_seq)
        gamma = np.zeros((self.num_hidden_states, num_observed_seq))
        digamma = np.zeros((self.num_hidden_states, self.num_hidden_states, num_observed_seq-1))

        gamma = alpha * beta / c

        for t in range(num_observed_seq - 1):
          o_t_plus_1 = self._get_observation_idx(observed_seq[t + 1])
          for i in range(self.num_hidden_states):
            digamma[i, :, t] = alpha[i, t] * self.A[i, :] * self.B[:, o_t_plus_1] * beta[:, t + 1]

        return gamma, digamma

    def re_estimate_A(self, gamma, digamma, observed_seq):
        num_observed_seq = len(observed_seq)
        newA = np.zeros((self.num_hidden_states, self.num_hidden_states))
        """
        use local digamma and gamma to update the local A, return a newA
        reference: https://github.com/vrjkmr/hmm/blob/master/hmm.py
        """
        
        newA = digamma.sum(axis=2) / gamma[:, :-1].sum(axis=1).reshape(-1,1)
        """ # works, but too slow
        for i in range(self.num_hidden_states):
          denom = 0
          for t in range(num_observed_seq-1):
            denom += gamma[i,t]

            for j in range(self.num_hidden_states):

              numer = 0

              for t in range(num_observed_seq-1):

                numer+= digamma[i,j,t]
                newA[i,j] = numer / denom
        """     

        return newA

    
    def re_estimate_B(self, gamma, observed_seq):
        num_observed_seq = len(observed_seq)
        newB = np.zeros((self.num_hidden_states, self.vocabulary_size))

        for idx, o in enumerate(self.vocabulary):
          indices = np.argwhere(observed_seq == o).flatten()
          newB[:, idx] = gamma[:, indices].sum(axis=1) / gamma.sum(axis=1)
        
        """
        M = self.B.shape[1]

        for i in range(self.num_hidden_states):
          denom = 0
          for t in range(num_observed_seq):
            denom += gamma[i,t]

          for j in range(M):
            numer = 0
            for t in range(num_observed_seq):
              o_t = self._get_observation_idx(observed_seq[t])
              if (o_t == j):
                numer += gamma[i,t]
            newB[i,j] = numer/denom
        """    
        return newB
    
    def re_estimate_pi(self, gamma):
      newpi = gamma[:, 0]
      return newpi


    # Return a "completed" sample by additing additional steps based on model probability.
    def complete_sequence(self, sample, steps):
        pass

    # Save the complete model to a file (most likely using np.save and pickles)
    def save_model(self, filename):
        model = {'num_hidden_state': self.num_hidden_states, 
                 'vocabulary':self.vocabulary,
                 'A': self.A, 
                 'B': self.B, 
                 'pi': self.pi}
        np.savez(filename, **model)

# Load a complete model from a file and return an HMM object (most likely using np.load and pickles)
def load_hmm(filename):
    num_hidden_states, vocabulary, A, B, pi = np.load(filename+'.npz', allow_pickle=True).values()
    hmm = HMM(num_hidden_states, vocabulary)
    hmm.A = A
    hmm.B = B
    hmm.pi = pi
    return hmm
    
# Load all the files in a subdirectory and return a giant list.
def load_subdir(path):
    data = []
    for filename in os.listdir(path):
        with open(os.path.join(path, filename)) as fh:
            data.append(fh.read())
    return data

def read_data(path):
  
    # Change the directory
    os.chdir(path)
      
    vocabulary = set()
    observations = [] 
    count = 0
    # iterate through all file
    for filename in os.listdir():
      with open(os.path.join(path, filename)) as fh:
        count += 1
        # can be more efficient
        sample = np.array(list(fh.read()))
        observations.append(sample)
        unique_chr_in_sample = set(sample)
        vocabulary.update(unique_chr_in_sample)
    print("finish reading {} files, dataset have {} unique characters".format(count, len(vocabulary)))
    return np.array(list(vocabulary)), observations

def graph_scores(LL_stack_list,hidden_states, hmm_model_name):
  
  _, line_chart = plt.subplots(figsize=(10, 10))
  line_chart.invert_yaxis()

  for i in range(hidden_states):
    y = LL_stack_list[i]
    x = range(len(y))
    Label = 'HiddenState ' + str(i+1)
    line_chart.plot(x, y, label=Label)
  plt.title('iteration vs log likelihood for ' + hmm_model_name)
  line_chart.legend()
  plt.show()

def has_outlier(observation, model_vocabulary):
  if set(observation).issubset(model_vocabulary):
    return False
  return True

def get_accuracy(pos_hmm, neg_hmm, pos_test_samples, neg_test_samples, model_vocabulary):
  correct = 0
  total = 0
  num_outlier_test_samples = 0


  for observation in pos_test_samples:
    if has_outlier(observation, model_vocabulary):
      continue 
    
    pos_LL = pos_hmm.LL_helper(observation)
    neg_LL = neg_hmm.LL_helper(observation)

    if (pos_LL == 0) or (neg_LL == 0):
        #print("Pos LL or neg LL = 0")
      continue
    if pos_LL > neg_LL:
      correct += 1  
    total += 1

  for observation in neg_test_samples:
    if has_outlier(observation, model_vocabulary):
      continue 
    pos_LL = pos_hmm.LL_helper(observation)
    neg_LL = neg_hmm.LL_helper(observation)

    if (pos_LL == 0) or (neg_LL == 0):
        continue

    if neg_LL > pos_LL:
      correct += 1
    total += 1

  print("%d/%d correct; accuracy %f"%(correct, total, correct/total))


def main():
    
    # python hmm.py --train_path /aclImdbNorm/train --dev_path /aclImdbNorm/test --max_iters 3 --hidden_states 3 --num_train_data 1500 --num_test_data 1000 --use_classifier True --initilize_method random --test_convergent True --fixed_hidden_states True
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')

    parser.add_argument('--dev_path', default=None, help='Path to development (i.e., testing) data.')
    parser.add_argument('--train_path', default=None, help='Path to the training data directory.')
    parser.add_argument('--max_iters', type=int, default=30, help='The maximum number of EM iterations (default 30)')
    # parser.add_argument('--model_out', default=None, help='Filename to save the final model.')
    parser.add_argument('--initilize_method', default='random', help='random or uniform')
    parser.add_argument('--num_train_data', type=int, default=1500, help='number of training data')
    parser.add_argument('--num_test_data', type=int, default=1000, help='number of training data')

    parser.add_argument('--hidden_states', type=int, default=10, help='The number of hidden states to use. (default 10)')
    parser.add_argument('--use_classifier', type=lambda x: bool(strtobool(x)), help='True means test classification accuracy on test set')
    parser.add_argument('--test_convergent', type=lambda x: bool(strtobool(x)), default=True, help='True means test convergent of EM algorithm')
    parser.add_argument('--fixed_hidden_states', type=lambda x: bool(strtobool(x)), default=False, help='True means use fixed number of hidden states')

    args = parser.parse_args()
    train_path = args.train_path
    dev_path = args.dev_path
    max_iters = args.max_iters
    num_hidden_states = args.hidden_states
    # model_out = args.model_out
    initilize_method = args.initilize_method
    num_train_data = args.num_train_data
    num_test_data = args.num_test_data
    use_classifier = args.use_classifier
    test_convergent = args.test_convergent
    fixed_hidden_states = args.fixed_hidden_states

    #1. load training, need to parse in command line input, build vocabulary and turn data into a list of numpy
    cwd = os.getcwd()
    pos_train_vocabulary, pos_train_observations = read_data(path=cwd+train_path+"/pos")
    neg_train_vocabulary, neg_train_observations = read_data(path=cwd+train_path+"/neg")

    pos_test_vocabulary, pos_test_observations = read_data(path=cwd+dev_path+"/pos")
    neg_test_vocabulary, neg_test_observations = read_data(path=cwd+dev_path+"/neg")
    
    os.chdir(cwd)

    #2. train HMM but for fixed hidden states
    if fixed_hidden_states:
      print("\nHIDDEN_STATES",num_hidden_states)

      pos_hmm = HMM(num_hidden_states, pos_train_vocabulary)
      pos_hmm.initialize_A_B_pi(initilize_method)
      
      print("Training Positive HMM")
      pos_hmm.fit(pos_train_observations[:num_train_data], max_iters)

    else:
      list_of_pos_LL = []
      list_of_neg_LL = []
      
      for HIDDEN_STATES in range(1,num_hidden_states+1):
        print("\nHIDDEN_STATES",HIDDEN_STATES)

        #initialize HMM
        pos_hmm = HMM(HIDDEN_STATES, pos_train_vocabulary)
        neg_hmm = HMM(HIDDEN_STATES, neg_train_vocabulary)
        pos_hmm.initialize_A_B_pi(initilize_method)
        neg_hmm.initialize_A_B_pi(initilize_method)
        
        tic = time.time()

        print("Training Positive HMM")
        pos_LL_list = pos_hmm.fit(pos_train_observations[:num_train_data], max_iters)
        print("\nTraining Negative HMM")
        neg_LL_list = neg_hmm.fit(neg_train_observations[:num_train_data], max_iters)

        list_of_pos_LL.append(pos_LL_list)
        list_of_neg_LL.append(neg_LL_list)

        toc = time.time()
        print("Time Elapsed for hidden state{}: ", {HIDDEN_STATES},  toc - tic)

        if use_classifier:
          intersect_vocabulary = set(pos_train_vocabulary.flatten()).intersection(set(neg_train_vocabulary.flatten()))
          get_accuracy(pos_hmm, neg_hmm, pos_test_observations[:num_test_data], neg_test_observations[:num_test_data], intersect_vocabulary)

      if test_convergent:
        graph_scores(list_of_pos_LL, num_hidden_states, 'pos')
        graph_scores(list_of_neg_LL, num_hidden_states, 'neg')

# going up report the test sample that passed/.
if __name__ == '__main__':
    main()