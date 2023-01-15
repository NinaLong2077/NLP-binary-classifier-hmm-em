# NLP-tone-classifier
A binary classifier on sequential data (50000 positive/negative IMDB reviews) using hidden Markov model (HMM) with Expectation Maximization algorithm.

Steps:

1. Train and save a hmm model on postive training set

2. Train and save a hmm model on negative training set

3. Test positive model on positive test data, test positive model on negative test data. Positive model should score higher accuracy on positive than on negative test data.

4. Same for negative model



Both HMM and EM are built from scratch with helps from reading [vrjkmr](https://github.com/vrjkmr/hmm/blob/master/hmm.py), [ducanhnguyen](https://github.com/ducanhnguyen/hidden-markov-model/blob/master/src/hmmd.py), [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) and [Daniel Jurafsky & James H. Martin](https://web.stanford.edu/~jurafsky/slp3/A.pdf).

