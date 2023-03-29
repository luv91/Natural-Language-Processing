

# refer to the written report to understand better:
    
# D:\computer science\Coding One Stop\NortheasternSubjects\Summer Projects\NLP_Transformers\Codes\cs224n_word_2_vec_assignment_2\a2_written_part


import argparse
import numpy as np
import random

from utils.gradcheck import gradcheck_naive, grad_tests_softmax, grad_tests_negsamp
from utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE (~1 Line)
    s = 1 / (1 + np.exp(-x))
    ### END YOUR CODE

    return s

"""
naiveSoftmaxLossAndGradient is a function that computes the 
naive softmax loss and gradients for a given center word and an 
outside word. It takes in four arguments:

centerWordVec: A numpy ndarray representing the embedding vector for
 the center word. Its shape is (word vector length, ), where
 "word vector length" refers to the dimensionality of the word embeddings.
 
outsideWordIdx: An integer representing the index of the outside
 word in the vocabulary. The index is used to select the corresponding
 row of the outsideVectors matrix, which contains the embedding vectors
 for all the words in the vocabulary.
 
outsideVectors: A numpy ndarray representing the embedding vectors 
for all the words in the vocabulary. Its shape is (num words in vocab
                                                   , word vector length).

dataset: An optional argument that is not used in this function, 
but is included for compatibility with other functions that use 
the same interface.

"""
def naiveSoftmaxLossAndGradient(
        centerWordVec,
        outsideWordIdx,
        outsideVectors,
        dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models. For those unfamiliar with numpy notation, note 
    that a numpy ndarray with a shape of (x, ) is a one-dimensional array, which
    you can effectively treat as a vector with length x.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (transpose of U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    ### YOUR CODE HERE (~6-8 Lines)

    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow.
    centerWordVec = centerWordVec[:, np.newaxis]  # (d,) -> (d, 1)
    
    #  y_hat is predicting the probability distribution of the 
    # outside words given the center word.
    """
    In other words, y_hat is predicting the likelihood that each 
    outside word is the correct context word for the given center word,
    and this prediction is used to compute the loss and gradients in the
    naive softmax loss function.
    """
    y_hat = softmax(np.matmul(outsideVectors, centerWordVec).T).T  # (V, 1)
    
    
    """
    In the line loss = -np.log(y_hat[outsideWordIdx]), loss is computed 
    as the negative log-likelihood of the true outside word given the center word.

    Recall that y_hat is a vector of probabilities representing the 
    likelihood of each outside word being the correct context word 
    given the center word. outsideWordIdx is the index of the true 
    outside word, i.e., the correct context word for the given center word. 
    
    y_hat[outsideWordIdx] therefore represents the probability assigned to 
    the true outside word by the softmax function. Taking the negative log 
    of this probability gives the log-likelihood of the true outside word.
    
    Negating it then gives the negative log-likelihood, which is used as
    the loss function in the naive softmax.
    """
    loss = -np.log(y_hat[outsideWordIdx])  # Negative-log likelihood, (V, 1)
    # Construct y here.
    y = np.zeros_like(y_hat)  # (V, 1)
    # Or define: delta = y_hat[outsideWordIdx] - 1, and use delta instead.
    
    """
    Recall that the softmax function takes as input a vector and 
    produces as output a probability distribution over a set of classes.
    In the case of the naive softmax loss, the classes are the vocabulary 
    words, and we want to predict the probability of each vocabulary word 
    given the center word.

    The one-hot vector y has the same length as y_hat and it has a 1 in the
    position corresponding to the index of the target word and 0 elsewhere.
    This one-hot vector is used to calculate the gradient of the loss with 
    respect to the outside word vectors, which is the third output of the function.
    
    """
    
    y[outsideWordIdx] = 1  # One-hot column vector.
    
    
    # Taken the gradient of the center and the outside vectors. 
    gradCenterVec = np.matmul(outsideVectors.T, y_hat - y)  # (d, V) * (V, 1) = (d, 1)
    gradOutsideVecs = np.matmul(y_hat - y, centerWordVec.T)  # (V, 1) * (1, d) = (V, d). Result transposed.

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def test_sigmoid():
    """ Test sigmoid function """
    print("=== Sanity check for sigmoid ===")
    assert sigmoid(0) == 0.5
    assert np.allclose(sigmoid(np.array([0])), np.array([0.5]))
    
    # True ==> 
    assert np.allclose(sigmoid(np.array([1, 2, 3])), np.array([0.73105858, 0.88079708, 0.95257413]))
    print("Tests for sigmoid passed!")
    
"""
This negative sampling function is designed to sample K negative words (indices) that are 
different from a given outside word index (outsideWordIdx). The outside word is a context
 word in the Skip-gram model of Word2Vec, which means it's one of the words surrounding 
 the target (center) word within a given window size.

Let's break down the function and explain its details:

outsideWordIdx: This is the index of the outside (context) word that we want to exclude 
from the negative samples.

dataset: This is an object representing the dataset being used, which has a method called 
sampleTokenIdx(). This method returns a random word index from the dataset's vocabulary,
 based on a pre-defined probability distribution (usually the unigram distribution).

K: This is the number of negative samples we want to generate.

The function initializes an empty list negSampleWordIndices of length K to store the negative samples.

It then iterates K times, sampling a new index (newidx) using the dataset.sampleTokenIdx() 
method. If newidx is equal to the given outsideWordIdx, the function keeps sampling until 
it gets a different index. This ensures that the negative samples don't include the context word.

Once a valid newidx is found, it is added to the negSampleWordIndices list.

After K iterations, the function returns the negSampleWordIndices list containing K negative 
samples that are different from the outsideWordIdx.

"""
def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices

def negSamplingLossAndGradient(
        centerWordVec,
        outsideWordIdx,
        outsideVectors,
        dataset,
        K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    
    """
    
    In the negSamplingLossAndGradient function, the line indices = [outsideWordIdx] + negSampleWordIndices 
    combines the index of the outside (context) word and the indices of the K negative samples 
    into a single list.

    The purpose of this line is to create a list of indices that includes both the true context 
    word (outsideWordIdx) and the negative samples (negSampleWordIndices) to compute the loss 
    and gradients for both positive and negative samples in one go.
    
    Remember that the goal of negative sampling is to maximize the probability of the
    true context word (positive sample) and minimize the probabilities of the negative
    samples. To achieve this, the loss and gradients are calculated for both the positive
    sample and the negative samples together.

    By combining the true context word index and the negative sample indices into
    a single list (indices), it becomes easier to perform the calculations for the
    loss and gradients in a vectorized manner. This helps speed up the computation
    and makes the code more efficient.
    
    
    """
    indices = [outsideWordIdx] + negSampleWordIndices

    ### YOUR CODE HERE (~10 Lines)

    ### Please use your implementation of sigmoid in here.

    centerWordVec = centerWordVec[:, np.newaxis]  # (d,) -> (d, 1)
    
    """
    The purpose of these two lines is to create an array sampleVec that contains
    the word vectors of both the true context word (positive sample) and the negative 
    samples, but with a sign change for the negative samples.

    Here's what each line does:
    
    sampleVec = -outsideVectors[indices]: This line creates an array sampleVec that
    contains the word vectors of all the words in indices (i.e., the true context word 
                                                           and the negative samples).
    
    However, it multiplies all of these vectors by -1. The reason for this is to differentiate
    between the positive sample (the true context word) and the negative samples, as we'll be 
    adding their contributions to the loss and gradient differently later on.
    
    sampleVec[0] *= -1: In the previous step, we multiplied all the word vectors by -1, 
    which means the word vector for the true context word (positive sample) is also 
    multiplied by -1. This line corrects that by multiplying the first element of sampleVec 
    (which corresponds to the true context word) by -1 again, effectively returning it to 
    its original value. Now, sampleVec has the word vector of the true context word at index 0, 
    followed by the word vectors of the negative samples, which are all multiplied by -1.
    
    """
    sampleVec = -outsideVectors[indices]
    sampleVec[0] *= -1  # (K+1, d)
    
    """
    
    The purpose of the line temp = sigmoid(np.matmul(sampleVec, centerWordVec)) - 1
    is to calculate the error terms that will be used to compute the loss and 
    gradients for the negative sampling loss function.
    
    ... - 1: We subtract 1 from the sigmoid values to get the error terms for the
    true context word and the negative samples. Since the first element in sampleVec
    corresponds to the true context word and its dot product with the center word vector
    is expected to be high, the sigmoid value for the true context word will be close 
    to 1, and subtracting 1 will result in a small error term. On the other hand, the
    dot products for the negative samples should be low (due to negation), resulting 
    in sigmoid values close to 0, and thus, their error terms will be close to -1 after subtraction.
    """
    # sigmoid part -==> in the loss function calculation. 
    temp = sigmoid(np.matmul(sampleVec, centerWordVec)) - 1  # (K+1, d) * (d, 1) -> (K+1, 1)
    
    """
    The line loss = -np.sum(np.log(temp + 1)) does the following:

    temp + 1: Adds 1 to the error terms in temp. This results in a value close to
    1 for the true context word and values close to 0 for the negative samples.
    
    np.log(...): Takes the natural logarithm of the values obtained in 
    the previous step. The logarithm of 1 is 0, and the logarithm of values
    close to 0 is a large negative number.
    
    -np.sum(...): Negates and sums the logarithm values calculated in the previous step
    to compute the final loss value. The negation ensures that the loss is a positive 
    value, as the logarithm of values close to 0 is negative. The sum aggregates
    the contributions of the true context word and the negative samples to the overall loss.
    """
    loss = -np.sum(np.log(temp + 1))  # This is a scalar!
    
    """
    The line gradCenterVec = np.matmul(sampleVec.T, temp) calculates the 
    gradient of the loss with respect to the center word vector (centerWordVec). 
    This gradient is needed to update the center word vector during the training process.
    
    The loss itself is calculated in the line loss = -np.sum(np.log(temp + 1)). 
    This line computes the negative sampling loss using the error terms stored in temp, 
    which were calculated earlier as sigmoid(np.matmul(sampleVec, centerWordVec)) - 1.
    
    Check out answer (i) of question g of question 1 in written report: a2_written_part.pdf , page 4
    """
    gradCenterVec = np.matmul(sampleVec.T, temp)  # (d, K+1) * (K+1, 1) -> (d, 1)
    
    
    gradOutsideVecs = np.zeros_like(outsideVectors)  # Initialize the gradient.
    # Calculate the gradient for the real context word and negative samples, respectively.
    
    # 
    """
    Check out answer (i) of question g of question 1.in written report: a2_written_part.pdf , page 4
    """
    grad = -np.matmul(temp, centerWordVec.T)
    grad[0] *= -1
    
    """
    np.add.at(gradOutsideVecs, indices, grad): This line uses the numpy.add.at 
    function to add the gradient updates in grad to the corresponding rows of the 
    gradOutsideVecs array specified by the indices list. The numpy.add.at function
    allows you to add the gradient updates in-place, which means that the gradOutsideVecs
    array is directly updated with the new gradient values.
    
    """
    np.add.at(gradOutsideVecs, indices, grad)  # (K+1, d), transposed.

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs

def getDummyObjects():
    """ Helper method for naiveSoftmaxLossAndGradient and negSamplingLossAndGradient tests """

    """
    dummySampleTokenIdx: This function returns a random integer between
    0 and 4 (inclusive). It's used to randomly sample a token index.
    """
    def dummySampleTokenIdx():
        return random.randint(0, 4)
    
    """
    getRandomContext: This function returns a tuple with a random token 
    from a list of tokens (["a", "b", "c", "d", "e"]) and a context list 
    of tokens. The context list is created by randomly selecting tokens from
    the same list, with length equal to 2 times the parameter C.
    
    The getDummyObjects function is a helper function that creates a simple, 
    controlled environment for testing other functions related to word embeddings,
    such as naiveSoftmaxLossAndGradient and negSamplingLossAndGradient.
    
    It returns a dummy dataset, word vectors, and a dictionary of tokens 
    (words) with their corresponding indices.
    
    
    can you explain with a simple example? say take a word 'hello' 
    and explain the utility of getRandomContext?
    
    Sure, let's say we have the word "hello" in a sentence:


    "Hello, how are you doing today?"
    To create a context for the word "hello", we can use the getRandomContext 
    function. Let's assume C is 2, which means we want to include the two words 
    to the left and two words to the right of the target word in the context.
    
    Here's an example of how getRandomContext might create a context for 
    the word "hello" in this sentence:
    
    makefile
    Copy code
    target_word = "hello"
    context_words = ["how", "are", "you", "doing"]
    In this example, "hello" is the target word, and the context consists 
    of the four words that appear two words to the left and two words to 
    the right of the target word.
    
    We can then use this tuple (target_word, context_words) as an input to the naiveSoftmaxLossAndGradient and negSamplingLossAndGradient functions to calculate the loss and gradient for the word embedding model. By randomly selecting different contexts for each target word, the model can learn to associate different meanings and usages of the target word with different contexts, which can improve the quality of the learned word embeddings.


    
    """
    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        
        print(" tokens[random.randint(0, 4)], \
               [tokens[random.randint(0, 4)] for i in range(2 * C)]", 
               tokens[random.randint(0, 4)], \
                     [tokens[random.randint(0, 4)] for i in range(2 * C)])
        return tokens[random.randint(0, 4)], \
               [tokens[random.randint(0, 4)] for i in range(2 * C)]

    dataset = type('dummy', (), {})()
    
    """
    So, to get a random token index, the dummySampleTokenIdx() function is
    called, which generates a random integer between 0 and 4 (inclusive) and 
    returns it. This random integer is used as an index to retrieve the 
    corresponding token from the dummy_tokens dictionary.
    
    """
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext
    

    random.seed(31415)
    np.random.seed(9265)
    
    #  the line dummy_vectors = normalizeRows(np.random.randn(10, 3)) 
    # is generating a set of word vectors. 
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    
    """
    The token index is a unique integer assigned to each word in the 
    vocabulary. In this particular implementation, the dummy_tokens 
    dictionary is used to map each token (word) to its corresponding index. 
    
    The keys of this dictionary are the words, and the values are the indices.
    
    """
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    print("dataset, dummy_vectors, dummy_tokens",dataset, dummy_vectors, dummy_tokens)
    return dataset, dummy_vectors, dummy_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test your implementations.')
    parser.add_argument('function', nargs='?', type=str, default='all',
                        help='Name of the function you would like to test.')

    args = parser.parse_args()
    if args.function == 'sigmoid':
        test_sigmoid()
        
    getDummyObjects()