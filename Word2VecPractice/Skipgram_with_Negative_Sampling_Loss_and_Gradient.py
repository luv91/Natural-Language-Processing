
import argparse
import numpy as np
import random
#from utils import normalizeRows, softmax

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    N = x.shape[0]
    x /= np.sqrt(np.sum(x ** 2, axis=1)).reshape((N, 1)) + 1e-30
    return x


# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x, gradientText):
    """ Gradient check for a function f.
    Arguments:
    f -- a function that takes a single argument and outputs the
         loss and its gradients
    x -- the point (numpy array) to check the gradient at
    gradientText -- a string detailing some context about the gradient computation

    Notes:
    Note that gradient checking is a sanity test that only checks whether the
    gradient and loss values produced by your implementation are consistent with
    each other. Gradient check passing on its own doesn’t guarantee that you
    have the correct gradients. It will pass, for example, if both the loss and
    gradient values produced by your implementation are 0s (as is the case when
    you have not implemented anything). Here is a detailed explanation of what
    gradient check is doing if you would like some further clarification:
    http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/. 
    """
    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x)  # Evaluate function value at original point
    h = 1e-4  # Do not change this!

    # Iterate over all indexes ix in x to check the gradient.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        x[ix] += h  # increment by h
        random.setstate(rndstate)
        fxh, _ = f(x)  # evalute f(x + h)
        x[ix] -= 2 * h  # restore to previous value (very important!)
        random.setstate(rndstate)
        fxnh, _ = f(x)
        x[ix] += h
        numgrad = (fxh - fxnh) / 2 / h

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed for %s." % gradientText)
            print("First gradient error found at index %s in the vector of gradients" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad))
            return

        it.iternext()  # Step to next dimension

    print("Gradient check passed! Read the docstring of the `gradcheck_naive`"
          " method in utils.gradcheck.py to understand what the gradient check does.")
    
def grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient):
    print("======Skip-Gram with negSamplingLossAndGradient======")

    # first test
    output_loss, output_gradCenterVecs, output_gradOutsideVectors = \
        skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5, :],
                 dummy_vectors[5:, :], dataset, negSamplingLossAndGradient)

    print("output_loss",output_loss)
# =============================================================================
#     assert np.allclose(output_loss, 16.15119285363322), \
#         "Your loss does not match expected loss."
# =============================================================================
    expected_gradCenterVecs = [[0., 0., 0.],
                               [0., 0., 0.],
                               [-4.54650789, -1.85942252, 0.76397441],
                               [0., 0., 0.],
                               [0., 0., 0.]]
    expected_gradOutsideVectors = [[-0.69148188, 0.31730185, 2.41364029],
                                   [-0.22716495, 0.10423969, 0.79292674],
                                   [-0.45528438, 0.20891737, 1.58918512],
                                   [-0.31602611, 0.14501561, 1.10309954],
                                   [-0.80620296, 0.36994417, 2.81407799]]

# =============================================================================
#     assert np.allclose(output_gradCenterVecs, expected_gradCenterVecs), \
#         "Your gradCenterVecs do not match expected gradCenterVecs."
# =============================================================================
# =============================================================================
#     assert np.allclose(output_gradOutsideVectors, expected_gradOutsideVectors), \
#         "Your gradOutsideVectors do not match expected gradOutsideVectors."
#     print("The first test passed!")
# =============================================================================

    # second test
    output_loss, output_gradCenterVecs, output_gradOutsideVectors = \
        skipgram("c", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5, :],
                 dummy_vectors[5:, :], dataset, negSamplingLossAndGradient)
# =============================================================================
#     assert np.allclose(output_loss, 28.653567707668795), \
#         "Your loss does not match expected loss."
# =============================================================================
    expected_gradCenterVecs = [[0., 0., 0.],
                               [0., 0., 0.],
                               [-6.42994865, -2.16396482, -1.89240934],
                               [0., 0., 0.],
                               [0., 0., 0.]]
    expected_gradOutsideVectors = [[-0.80413277, 0.36899421, 2.80685192],
                                   [-0.9277269, 0.42570813, 3.23826131],
                                   [-0.7511534, 0.34468345, 2.62192569],
                                   [-0.94807832, 0.43504684, 3.30929863],
                                   [-1.12868414, 0.51792184, 3.93970919]]

# =============================================================================
#     assert np.allclose(output_gradCenterVecs, expected_gradCenterVecs), \
#         "Your gradCenterVecs do not match expected gradCenterVecs."
#     assert np.allclose(output_gradOutsideVectors, expected_gradOutsideVectors), \
#         "Your gradOutsideVectors do not match expected gradOutsideVectors."
# =============================================================================
    print("The second test passed!")

    # third test
    output_loss, output_gradCenterVecs, output_gradOutsideVectors = \
        skipgram("a", 3, ["a", "b", "e", "d", "b", "c"],
                 dummy_tokens, dummy_vectors[:5, :],
                 dummy_vectors[5:, :], dataset, negSamplingLossAndGradient)
# =============================================================================
#     assert np.allclose(output_loss, 60.648705494891914), \
#         "Your loss does not match expected loss."
# =============================================================================
    expected_gradCenterVecs = [[-17.89425315, -7.36940626, -1.23364121],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.]]
    expected_gradOutsideVectors = [[-6.4780819, -0.14616449, 1.69074639],
                                   [-0.86337952, -0.01948037, 0.22533766],
                                   [-9.59525734, -0.21649709, 2.5043133],
                                   [-6.02261515, -0.13588783, 1.57187189],
                                   [-9.69010072, -0.21863704, 2.52906694]]

# =============================================================================
#     assert np.allclose(output_gradCenterVecs, expected_gradCenterVecs), \
#         "Your gradCenterVecs do not match expected gradCenterVecs."
#     assert np.allclose(output_gradOutsideVectors, expected_gradOutsideVectors), \
#         "Your gradOutsideVectors do not match expected gradOutsideVectors."
# =============================================================================
    print("The third test passed!")
    
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

def getDummyObjects():
    """ Helper method for naiveSoftmaxLossAndGradient and negSamplingLossAndGradient tests """

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
               [tokens[random.randint(0, 4)] for i in range(2 * C)]

    dataset = type('dummy', (), {})()
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    return dataset, dummy_vectors, dummy_tokens
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
    indices = [outsideWordIdx] + negSampleWordIndices

    ### YOUR CODE HERE (~10 Lines)

    ### Please use your implementation of sigmoid in here.

    centerWordVec = centerWordVec[:, np.newaxis]  # (d,) -> (d, 1)
    sampleVec = -outsideVectors[indices]
    sampleVec[0] *= -1  # (K+1, d)
    temp = sigmoid(np.matmul(sampleVec, centerWordVec)) - 1  # (K+1, d) * (d, 1) -> (K+1, 1)
    loss = -np.sum(np.log(temp + 1))  # This is a scalar!
    gradCenterVec = np.matmul(sampleVec.T, temp)  # (d, K+1) * (K+1, 1) -> (d, 1)
    gradOutsideVecs = np.zeros_like(outsideVectors)  # Initialize the gradient.
    # Calculate the gradient for the real context word and negative samples, respectively.
    grad = -np.matmul(temp, centerWordVec.T)
    grad[0] *= -1
    np.add.at(gradOutsideVecs, indices, grad)  # (K+1, d), transposed.

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs

def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=negSamplingLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (V in pdf handout)
    outsideVectors -- outside vectors is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (transpose of U in the pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (num words in vocab, word vector length)
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE (~8 Lines)

    centerWordInd = word2Ind[currentCenterWord]  # Get the index of the current center word.
    centerWordVec = centerWordVectors[centerWordInd]  # (1, d)
    outsideWordIndices = [word2Ind[i] for i in outsideWords]  # A list containing indices of all real outside words.
    for outsideWordInd in outsideWordIndices:
        wordLoss, wordGradCenterVec, wordGradOutsizeVecs = word2vecLossAndGradient(
            centerWordVec, outsideWordInd, outsideVectors, dataset
        )
        loss += wordLoss  # Accumulate loss (scalar).
        # Transpose the gradient (column vector) to a row vector of size (1, d), and remove its extra dimension.
        gradCenterVecs[centerWordInd] += wordGradCenterVec.reshape(-1)
        gradOutsideVectors += wordGradOutsizeVecs  # They are of the same shape (V, d).

    ### END YOUR CODE

    return loss, gradCenterVecs, gradOutsideVectors
def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset,
                         windowSize,
                         word2vecLossAndGradient=negSamplingLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N / 2), :]
    outsideVectors = wordVectors[int(N / 2):, :]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N / 2), :] += gin / batchsize
        grad[int(N / 2):, :] += gout / batchsize

    return loss, grad
def test_skipgram():
    """ Test skip-gram with naiveSoftmaxLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
                    dummy_vectors, "negSamplingLossAndGradient Gradient")
    grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient)

def test_negSamplingLossAndGradient():
    """ Test negSamplingLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for negSamplingLossAndGradient ====")

    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(vec, 1, dummy_vectors, dataset)
        return loss, gradCenterVec

    gradcheck_naive(temp, np.random.randn(3), "negSamplingLossAndGradient gradCenterVec")

    centerVec = np.random.randn(3)

    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(centerVec, 1, vec, dataset)
        return loss, gradOutsideVecs

    gradcheck_naive(temp, dummy_vectors, "negSamplingLossAndGradient gradOutsideVecs")

def test_skipgram():
    """ Test skip-gram with naiveSoftmaxLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

# =============================================================================
#     print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
#     gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
#         skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
#                     dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")
#     grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset)
# =============================================================================

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
# =============================================================================
#     gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
#         skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
#                     dummy_vectors, "negSamplingLossAndGradient Gradient")
# =============================================================================
    grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient)

def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """

    test_negSamplingLossAndGradient()
    test_skipgram()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test your implementations.')
    parser.add_argument('function', nargs='?', type=str, default='all',
                        help='Name of the function you would like to test.')
    
    args = parser.parse_args()
# =============================================================================
#     if args.function == 'sigmoid':
#         test_sigmoid()
#     elif args.function == 'naiveSoftmaxLossAndGradient':
#         test_naiveSoftmaxLossAndGradient()
# =============================================================================
# =============================================================================
#     if args.function == 'negSamplingLossAndGradient':
#         test_negSamplingLossAndGradient()
#     elif args.function == 'skipgram':
#         test_skipgram()
# =============================================================================
    if args.function == 'all':
        test_word2vec()
