"""EECS 445 - Winter 2023.

Project 1
"""

import pandas as pd
import numpy as np
import itertools
import string
import nltk as nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib import pyplot as plt
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
#from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 




from helper import *

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)



def extract_word(input_string):
    """Preprocess review into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along whitespace.
    Return the resulting array.

    E.g.
    > extract_word("I love EECS 445. It's my favorite course!")
    > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]

    Input:
        input_string: text for a single review
    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    
    result = string.punctuation
    s = str()
    s = input_string

    for i in range(0, len(input_string)):
        test = input_string[i]
        if str(s[i]).isalpha():
            s = s.replace(s[i], s[i].lower())
        elif (not(str(s[i]).isnumeric()) and s[i] in result):
            s = s.replace(s[i], " ")

    
    final = s.split()

    return final


def extract_word2(input_string):
    """Preprocess review into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along whitespace.
    Return the resulting array.

    E.g.
    > extract_word("I love EECS 445. It's my favorite course!")
    > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]

    Input:
        input_string: text for a single review
    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    
    result = string.punctuation
    s = str()
    s = input_string

    for i in range(0, len(input_string)):
        test = input_string[i]
        if str(s[i]).isalpha():
            s = s.replace(s[i], s[i].lower())
        elif (not(str(s[i]).isnumeric()) and s[i] in result):
            s = s.replace(s[i], " ")

    
    final = s.split()

    stemmedWords = []
    ps = PorterStemmer()
    for w in final:
        root = ps.stem(w)
        stemmedWords.append(root)

    lemmatizer = WordNetLemmatizer()
    lemmedWords = []
    for word in stemmedWords:
        lemmedWords.append(lemmatizer.lemmatize(word))
    
    return lemmedWords



def extract_dictionary(df):
    """Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was
    found).

    E.g., with input:
        | text                          | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

    The output should be a dictionary of indices ordered by first occurence in
    the entire dataset:
        {
           it: 0,
           was: 1,
           the: 2,
           best: 3,
           of: 4,
           times: 5,
           blurst: 6
        }
    The index should be autoincrementing, starting at 0.

    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}
    # TODO: Implement this function
    #should this be a loop through df, what is df, iterarows
    count = 0



    for index, row in df.iterrows():
        temp = row["text"]
        words = extract_word(row["text"])
        for j in words:
            if not(j in word_dict.keys()):
                word_dict[j] = count
                count = count + 1
    return word_dict

def extract_dictionary2(df):
    """Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was
    found).

    E.g., with input:
        | text                          | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

    The output should be a dictionary of indices ordered by first occurence in
    the entire dataset:
        {
           it: 0,
           was: 1,
           the: 2,
           best: 3,
           of: 4,
           times: 5,
           blurst: 6
        }
    The index should be autoincrementing, starting at 0.

    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}
    # TODO: Implement this function
    #should this be a loop through df, what is df, iterarows
    count = 0



    for index, row in df.iterrows():
        temp = row["text"]
        words = extract_word2(row["text"])
        for j in words:
            if not(j in word_dict.keys()):
                word_dict[j] = count
                count = count + 1
    return word_dict


def generate_feature_matrix(df, word_dict):
    """Create matrix of feature vectors for dataset.

    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review.  Use the word_dict to find the
    correct index to set to 1 for each place in the feature vector. The
    resulting feature matrix should be of dimension (# of reviews, # of words
    in dictionary).

    Input:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
        

    """
    
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))

    # word_dict = extract_dictionary(df)
    # for index, row in df.iterrows():

    rowCounter = 0
    wordColCounter = 0

    for index, row in df.iterrows():

        for temps in extract_word(row["text"]):
            # check = word_dict[0][0]
            #check if it even exists
            if temps in word_dict.keys():
                temp = word_dict[str(temps)]
                feature_matrix[rowCounter][temp] = 1
        rowCounter = rowCounter + 1

    # TODO: Implement this function
    return feature_matrix


def generate_feature_matrix2(df, word_dict):
    """Create matrix of feature vectors for dataset.

    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review.  Use the word_dict to find the
    correct index to set to 1 for each place in the feature vector. The
    resulting feature matrix should be of dimension (# of reviews, # of words
    in dictionary).

    Input:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
        

    """
    
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))

    # word_dict = extract_dictionary(df)
    # for index, row in df.iterrows():

    rowCounter = 0
    wordColCounter = 0

    for index, row in df.iterrows():

        for temps in extract_word2(row["text"]):
            # check = word_dict[0][0]
            #check if it even exists
            if temps in word_dict.keys():
                temp = word_dict[str(temps)]
                feature_matrix[rowCounter][temp] = 1
        rowCounter = rowCounter + 1

    # TODO: Implement this function
    return feature_matrix



def performance(y_true, y_pred, metric="accuracy"):
    return metrics.accuracy_score(y_true, y_pred)

def performancePrecision(y_true, y_pred, metric="precision"):
    return metrics.precision_score(y_true, y_pred)

def performanceF1(y_true, y_pred, metric="f1_score"):
    return metrics.f1_score(y_true, y_pred)

def performanceSpec(y_true, y_pred, metric="specificity"):
    # tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, label = [1,-1]).ravel()
    #svm_sign

    conf = metrics.confusion_matrix(y_true, y_pred, labels = [1,-1])
    TN = conf[1][1]
    FP = conf[1][0]

    return TN / (TN + FP)

def performanceSense(y_true, y_pred, metric="sensitivity"):
    return metrics.recall_score(y_true, y_pred)

def performanceAUROC(y_true, y_pred, metric = "AUROC"):
    return metrics.roc_auc_score(y_true, y_pred)
    

def performance(y_true, y_pred, metric="accuracy"):
    return metrics.accuracy_score(y_true, y_pred)





    """Calculate performance metrics.

    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.

    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    # TODO: Implement this function
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.



def cv_performance(clf, X, y, k=5, metric = ""):
    """Split data into k folds and run cross-validation.

    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64



    """

    scores = []
    # TODO: Implement this function
    # HINT: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful

         # LinearSVC.fit(train_index, test_index)
        # LinearSVC.predict(train_index)
        # LinearSVC.decision_function(train_index)
    #random_state=None
    skf = StratifiedKFold(n_splits=k, shuffle=False)
    #enumerate here?
    for train_index, test_index in skf.split(X,y):
    # for k, (train_index, test_index) in enumerate(skf.split(X,y)): decision only for aroc
        X_train, Y_train, X_test, Y_test = X[train_index], y[train_index], X[test_index], y[test_index]

        #clf.fit(X[train_index], y[train_index])
        clf.fit(X_train, Y_train)
        if metric == "AUROC":
            scores.append(performanceAUROC(y[test_index], clf.decision_function(X[test_index]), metric = "AUROC"))
        elif metric == "precision":
            scores.append(performancePrecision(y[test_index], clf.predict(X[test_index]), metric = "precision"))
        elif metric == "sensitivity":
            scores.append(performanceSense(y[test_index], clf.predict(X[test_index]), metric = "sensitivity"))
        elif metric == "specificity":
            scores.append(performanceSpec(y[test_index], clf.predict(X[test_index]), metric = "specificity"))
        elif metric == "f1_score":
            scores.append(performanceF1(y[test_index], clf.predict(X[test_index]), metric = "f1_score"))
        else:
            scores.append(performance(y[test_index], clf.predict(X[test_index]), metric = "accuracy"))


    # Put the performance of the model on each fold in the scores array
    # scores = []
    return np.array(scores).mean()


def select_param_linear(
    X, y, k=5, metric="", C_range=[], loss="hinge", penalty="l2", dual=True
):
    
    """Search for hyperparameters from the given candidates of linear SVM with 
    best k-fold CV performance.

    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge",
             other option of "squared_hinge")
        penalty: string specifying the penalty type used (default="l2",
             other option of "l1")
        dual: boolean specifying whether to use the dual formulation of the
             linear SVM (set True for penalty "l2" and False for penalty "l1"ß)
    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    # TODO: Implement this function
    # HINT: You should be using your cv_performance function here
    # to evaluate the performance of each SVM
    minimum = 0
    bestC = 0

    #AROC USE DECISION FUNCTION MUST COME BACK AND DO

    for stats in C_range:
        uses = LinearSVC(loss="hinge", penalty="l2", dual=True, C = stats, random_state=445)
        performances = cv_performance(uses, X, y, k=5, metric = metric)
        if minimum < performances:
            minimum = performances
            bestC = stats
    return bestC
        


    


def plot_weight(X, y, penalty, C_range, loss, dual):
    """Create a plot of the L0 norm learned by a classifier for each C in C_range.

    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        penalty: string for penalty type to be forwarded to the LinearSVC constructor
        C_range: list of C values to train a classifier on
        loss: string for loss function to be forwarded to the LinearSVC constructor
        dual: whether to solve the dual or primal optimization problem, to be
            forwarded to the LinearSVC constructor
    Returns: None
        Saves a plot of the L0 norms to the filesystem.
    """
    norm0 = []
    # TODO: Implement this part of the function
    # Here, for each value of c in C_range, you should
    # append to norm0 the L0-norm of the theta vector that is learned
    # when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)
    #.coef theta[0] norm = 0


    for item in C_range:
        clf = LinearSVC(penalty = penalty, loss = loss, dual = dual, random_state = 445, C = item)
        clf.fit(X, y)
        theta = clf.coef_[0]
        #finding L0
        store = np.linalg.norm(theta, ord = 0)
        norm0.append(store)
    

    

    plt.plot(C_range, norm0)
    #plt.xlim([0.001, 1])

    plt.xscale("log")
    plt.legend(["L0-norm"])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title("Norm-" + penalty + "_penalty.png")
    plt.savefig("Norm-" + penalty + "_penalty.png")
    plt.close()



def select_param_quadratic(X, y, k=5, metric="f1_score", param_range=[]):
    """Search for hyperparameters from the given candidates of quadratic SVM 
    with best k-fold CV performance.

    Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
        param_range: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for r. Each row of this array thus
            represents a pair of parameters to be tried together.
    Returns:
        The parameter values for a quadratic-kernel SVM that maximize
        the average 5-fold CV performance as a pair (C,r)

        predict want things to be positive
        call confusion matrix for specificity and sensitivity
        

        quadratic approach generating random C: np.random.uniform

        grid is nested for loop
        other only needs one for loop, two arrays sized 25 just match them
        plot weight 

        random_search don't generate grid, generate 25 values of c and 25 values of r and match


    """
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    best_C_val, best_r_val = 0.0, 0.0
    compare = 0


    for row in range(0, len(param_range)):
        use = SVC(kernel='poly', degree = 2, C = param_range[row][0], coef0 = param_range[row][1], gamma = 'auto')
        compares = cv_performance(use, X, y, k = 5, metric = metric)
        if(compares > compare):
            compare = compares
            best_C_val = param_range[row][0]
            best_r_val = param_range[row][1]
    return best_C_val, best_r_val



def Q3(X_train):
    dataframe = load_data("data/dataset.csv")
    print("3a: ")
    print(extract_word("It's a test sentance! Does it look CORRECT?"))

    print("3b: ")
    print(len(extract_dictionary(dataframe)))

    print("3c: still need to do")
    generate_feature_matrix(dataframe, X_train)



# def Q43(X_train, Y_train, X_test, Y_test):
#     Carr = [10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3]
#     Rarr = [10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3]
#     param_range = []

#     for C_val in Carr:
#         for R_val in Rarr:
#             param_range.append((C_val, R_val))



#     values = select_param_quadratic(X_train, Y_train, k = 5, metric = "AUROC", param_range=param_range)

#     print("UPDATED QUAD values: ")
#     print(values)

#     updatedQuad = SVC(C = values[0], coef0= values[1], kernel="poly", degree = 2, gamma = "auto")
#     updatedQuad.fit(X_train, Y_train)
#     x = updatedQuad.decision_function(X_test)
#     print("UPDATED QUAD PERFORMANCE")
#     print(performanceAUROC(Y_test, x, metric = "AUROC"))
    

    


#     # for C_val in range(0, len(Carr) - 1) :
#     #     for R_val in range(0, len(Rarr) - 1):
#     #         param_range[C_val][R_val] = (Carr[C_val], Rarr[R_val])


#     print("GRID METHOD QUAD param selection:")
#     dobs = select_param_quadratic(X_train, Y_train, k = 5, metric = "AUROC", param_range = param_range)
#     print(dobs)
    
#     print("Grid performance: ")
#     quads2 = SVC(kernel='poly', degree = 2, C = dobs[0], coef0 = dobs[1], gamma = 'auto', random_state=445)
#     print(cv_performance(quads2, X_train, Y_train, metric = "AUROC"))

    
#    # print(cv_performance(quads, X_train, Y_train, metric = "AUROC"))
#     #print(select_param_quadratic(X_train, Y_train, k=5, metric="accuracy", param_range=param_range))
    

#     param_range2 = []
#     Carr2 = list(np.random.uniform(low = -2, high = 3, size = 25))
#     Rarr2 = list(np.random.uniform(low = -2, high = 3, size = 25))


#     for C_val2 in range(0, len(Carr2)):
#         param_range2.append((10 ** Carr2[C_val2], 10 ** Rarr2[C_val2]))


#     print("RANDOM METHOD AUROC param selection:")
#     dobs2 = select_param_quadratic(X_train, Y_train, k = 5, metric = "AUROC", param_range = param_range2)
#     print(dobs2)
#     print("Random method AUROC performance:")
#     quads = SVC(kernel='poly', degree = 2, C = dobs2[0], coef0 = dobs2[1], gamma = 'auto', random_state=445)
#     print(cv_performance(quads, X_train, Y_train, metric = "AUROC"))


def Q3(X_train):
    print(extract_word("It's a test sentance! Does it look CORRECT?"))

    dataframe = load_data("data/dataset.csv")
    my_dict = extract_dictionary(dataframe)
    print(len(my_dict))

    #Need help on this one
    my_dict = extract_dictionary(dataframe)
    feat = generate_feature_matrix(dataframe, my_dict)

    print("AVERAGE NONZERO: ", np.sum(X_train) / len(X_train))

    bestWord = np.argmax(np.sum(X_train, axis = 0))
    print("Best word: ")
    print(list(my_dict.keys())[list(my_dict.values()).index(bestWord)])



    
    print(np.nonzero(feat))
    print(len(feat))
    print(len(np.nonzero(feat)))



def Q41b(X_train, Y_train):
    ts = [10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3]
    accuracy_CV_C = []
    precision_CV_C = []
    f1_CV_C = []
    sensitivity_CV_C = []
    specificity_CV_C = []
    AUROC_CV_Cs = []

    for item in ts:
        clf2 = LinearSVC(random_state = 445, C = item, loss= "hinge")
        accuracy_CV_C.append(cv_performance(clf2, X_train, Y_train, k=5, metric = "accuracy"))

    for item in ts:
        clf2 = LinearSVC(random_state = 445, C = item, loss= "hinge")
        precision_CV_C.append(cv_performance(clf2, X_train, Y_train, k=5, metric = "precision"))
    
    for item in ts:
        clf2 = LinearSVC(random_state = 445, C = item, loss= "hinge")
        f1_CV_C.append(cv_performance(clf2, X_train, Y_train, k=5, metric = "f1_score"))
    
    for item in ts:
        clf2 = LinearSVC(random_state = 445, C = item, loss= "hinge")
        sensitivity_CV_C.append(cv_performance(clf2, X_train, Y_train, k=5, metric = "sensitivity"))
    
    for item in ts:
        clf2 = LinearSVC(random_state = 445, C = item, loss= "hinge")
        specificity_CV_C.append(cv_performance(clf2, X_train, Y_train, k=5, metric = "specificity"))
    
    for item in ts:
        clf2 = LinearSVC(random_state = 445, C = item, loss= "hinge")
        AUROC_CV_Cs.append(cv_performance(clf2, X_train, Y_train, k=5, metric = "AUROC"))

    print("ACCURACY:")
    for items in accuracy_CV_C:
        print(items)
    print("final accuracy")
    print(select_param_linear(X = X_train, y = Y_train, k = 5, metric = "accuracy", C_range = ts, loss = "hinge", dual = True))
    print("PRECISION")
    for items in precision_CV_C:
        print(items)
    print("final precision")
    print(select_param_linear(X = X_train, y = Y_train, k = 5, metric = "precision", C_range = ts, loss = "hinge", dual = True))
    print("F1")
    for items in f1_CV_C:
        print(items)
    print("final f1")
    print(select_param_linear(X = X_train, y = Y_train, k = 5, metric = "f1_score", C_range = ts, loss = "hinge", dual = True))
    print("SENSITIVITY:")
    for items in sensitivity_CV_C:
        print(items)
    print("final sensitivity")
    print(select_param_linear(X = X_train, y = Y_train, k = 5, metric = "sensitivity", C_range = ts, loss = "hinge", dual = True))
    print("SPECIFICITY")
    for items in specificity_CV_C:
        print(items)
    print("final specificity")
    print(select_param_linear(X = X_train, y = Y_train, k = 5, metric = "specificity", C_range = ts, loss = "hinge", dual = True))
    print("AUROC")
    for items in AUROC_CV_Cs:
        print(items)
    print("final AUROC")
  
    print(select_param_linear(X = X_train, y = Y_train, k = 5, metric = "AUROC", C_range = ts, loss = "hinge", dual = True))



def Q41c(X_train, Y_train, X_test, Y_test):
    partC = LinearSVC(random_state = 445, C = 1, loss= "hinge")
    partC.fit(X_train, Y_train)
    
    print("Accuracy on TEST 3C")
    print(cv_performance(partC, X_test, Y_test, k = 5, metric = "Accuracy"))

    print("Precision on TEST 3C")
    print(cv_performance(partC, X_test, Y_test, k = 5, metric = "precision"))

    print("F1 on TEST 3C")
    print(cv_performance(partC, X_test, Y_test, k = 5, metric = "f1_score"))

    print("AUROC on TEST 3C")
    print(cv_performance(partC, X_test, Y_test, k = 5, metric = "AUROC"))

    print("Sensitivity on TEST 3C")
    print(cv_performance(partC, X_test, Y_test, k = 5, metric = "Sensitivity"))

    print("Specificity on TEST 3C")
    print(cv_performance(partC, X_test, Y_test, k = 5, metric = "Specificity"))


def Q42(X_train, Y_train, X_test, Y_test):
    Q42Val = select_param_linear(X_train,Y_train, k = 5, metric = "AUROC", C_range=[0.001, 0.01, 0.1, 1], loss = "squared_hinge", dual = False)
    print("C values: ")
    print(Q42Val)
    

    Q42Lin = LinearSVC(penalty="l1", loss = "squared_hinge", C = Q42Val, dual = False, random_state = 445)
    
    print("mean CV AUROC score: ")
    print(cv_performance(Q42Lin, X_train, Y_train, k = 5, metric = "AUROC"))

    Q42Lins = LinearSVC(penalty="l1", loss = "squared_hinge", C = Q42Val, dual = False, random_state = 445)
    Q42Lins.fit(X_train, Y_train)
    print("AUROC score on test set: ")
    print(performanceAUROC(Y_train, Q42Lins.decision_function(X_train), metric = "AUROC"))


def updatedQ43(X_train, Y_train, X_test, Y_test):
    Carr = [10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3]
    Rarr = [10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3]
    param_range = []

    for C_val in Carr:
        for R_val in Rarr:
            param_range.append((C_val, R_val))


    values = select_param_quadratic(X_train, Y_train, k = 5, metric = "AUROC", param_range=param_range)

    print("UPDATED QUAD values: ")
    print(values)

    updatedQuad = SVC(C = values[0], coef0= values[1], kernel="poly", degree = 2, gamma = "auto", random_state=445)
    updatedQuad.fit(X_train, Y_train)
    x = updatedQuad.decision_function(X_test)
    print("UPDATED QUAD PERFORMANCE")
    print(performanceAUROC(Y_test, x, metric = "AUROC"))



    param_range2 = []
    Carr2 = list(np.random.uniform(low = -2, high = 3, size = 25))
    Rarr2 = list(np.random.uniform(low = -2, high = 3, size = 25))


    for C_val2 in range(0, len(Carr2)):
        param_range2.append((10 ** Carr2[C_val2], 10 ** Rarr2[C_val2]))

    valuesRan = select_param_quadratic(X_train, Y_train, k = 5, metric = "AUROC", param_range=param_range2)
    print("UPDATED RANDOM values: ")
    print(valuesRan)

    updatedQuadRan = SVC(C = valuesRan[0], coef0= valuesRan[1], kernel="poly", degree = 2, gamma = "auto", random_state=445)
    updatedQuadRan.fit(X_train, Y_train)
    x = updatedQuadRan.decision_function(X_test)
    print("UPDATED RANDOM PERFORMANCE")
    print(performanceAUROC(Y_test, x, metric = "AUROC"))

def Q51(X_train, Y_train, X_test, Y_test):
    Q51data = LinearSVC(penalty= "l2", loss = "hinge", C = 0.01, class_weight= {-1: 1, 1 :10}, random_state=445)
    Q51data.fit(X_train, Y_train)

    print("Accuracy: ")
    print(performance(Y_test, Q51data.predict(X_test), "Accuracy"))

    print("AUROC: ")
    print(performanceAUROC(Y_test, Q51data.decision_function(X_test), "AUROC"))

    print("F1: ")
    print(performanceF1(Y_test, Q51data.predict(X_test), "f1_score"))

    print("Precision: ")
    print(performancePrecision(Y_test, Q51data.predict(X_test), "precision"))

    print("Sensitivity: ")
    print(performanceSense(Y_test, Q51data.predict(X_test), "sensitivity"))

    print("Specificity: ")
    print(performanceSpec(Y_test, Q51data.predict(X_test), "specificity"))

def Q52(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels):

    print("Length of IMB_features :", len(IMB_features), "Length of labels: ", len(IMB_labels))
    print("Length of IMB_test features: ", len(IMB_test_features), "Length of IMB_test_labels", len(IMB_test_labels))


    Q52data = LinearSVC(penalty = "l2", loss = "hinge", C = 0.01, class_weight={-1:1, 1:1}, random_state=445)
    Q52data.fit(IMB_features, IMB_labels)


    print("Length of Prediciton: ", len(Q52data.predict(IMB_test_features)))
    print("Length of Real: ", len(IMB_test_labels))
   

    print("Accuracy: ")
    print(performance(IMB_test_labels, Q52data.predict(IMB_test_features), "Accuracy"))

    print("AUROC: ")
    print(performanceAUROC(IMB_test_labels, Q52data.decision_function(IMB_test_features), "AUROC"))

    print("F1: ")
    print(performanceF1(IMB_test_labels, Q52data.predict(IMB_test_features), "f1_score"))

    print("Precision: ")
    print(performancePrecision(IMB_test_labels, Q52data.predict(IMB_test_features), "precision"))

    print("Sensitivity: ")
    print(performanceSense(IMB_test_labels, Q52data.predict(IMB_test_features), "sensitivity"))

    print("Specificity: ")
    print(performanceSpec(IMB_test_labels, Q52data.predict(IMB_test_features), "specificity"))


def Q53(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels):

    Q53data2 = LinearSVC(penalty = "l2", loss = "hinge", C = 0.01, class_weight={-1:5, 1:2}, random_state=445)
    Q53data2.fit(IMB_features, IMB_labels)

    vals = [0.01, 0.1, 0.2, 0.5, 0.7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    finalAccWeights = [0,0]
    maxAcc = 0

    finalPrecWeights = [0,0]
    maxPrec = 0

    finalF1Weights = [0,0]
    maxF1 = 0

    finalAurWeights = [0,0]
    maxAur = 0

    finalSpecWeights = [0,0]
    maxSpec = 0

    finalSensWeights = [0,0]
    maxSens = 0

    
    print("THIS IS IMPORTANT: ")
    Q53data = LinearSVC(penalty = "l2", loss = "hinge", C = 0.01, class_weight={-1:5, 1:2}, random_state=445)
    Q53data.fit(IMB_features, IMB_labels)
    print(cv_performance(Q53data, IMB_features, IMB_labels, k = 5, metric = "precision"))

    Q53dataComp = LinearSVC(penalty = "l2", loss = "hinge", C = 0.01, class_weight={-1:0.1, 1:0.25}, random_state=445)
    Q53dataComp.fit(IMB_features, IMB_labels)
    
    for i in vals:
        for j in vals:
            Q53data = LinearSVC(penalty = "l2", loss = "hinge", C = 0.01, class_weight={-1:i, 1:j}, random_state=445)
            Q53data.fit(IMB_features, IMB_labels)
            tempAcc = cv_performance(Q53data, IMB_features, IMB_labels, k = 5, metric = "accuracy")
            # tempAur = cv_performance(Q53data, IMB_features, IMB_labels, k = 5, metric = "AUROC")
            # tempPrec = cv_performance(Q53data, IMB_features, IMB_labels, k = 5, metric = "precision")
            # tempSense = cv_performance(Q53data, IMB_features, IMB_labels, k = 5, metric = "sensitivity")
            # tempSpec = cv_performance(Q53data, IMB_features, IMB_labels, k = 5, metric = "specificity")
            # tempF1 = cv_performance(Q53data, IMB_features, IMB_labels, k = 5, metric = "f1_score")
            if tempAcc > maxAcc:
                maxAcc = tempAcc
                finalAccWeights = [i , j]
            
            # if tempPrec > maxPrec:
            #     maxPrec = tempPrec
            #     finalPrecWeights = [i , j]
            
            # if tempF1 > maxF1:
            #     maxF1 = tempF1
            #     finalF1Weights = [i , j]

            # if tempAur > maxAur:
            #     maxAur = tempAur
            #     finalAurWeights = [i , j]
            
            # if tempSense > maxSens:
            #     maxSens = tempSense
            #     finalSensWeights = [i , j]
            
            # if tempSpec > maxSpec:
            #     maxSpec = tempSpec
            #     finalSpecWeights = [i , j]

    Q53data = LinearSVC(penalty = "l2", loss = "hinge", C = 0.01, class_weight={-1:finalAccWeights[0], 1:finalAccWeights[1]}, random_state=445)
    Q53data.fit(IMB_features, IMB_labels)        

    print("Accuracy: ")
    print(maxAcc)
    print(finalAccWeights)

    print("AUROC: ", performanceAUROC(IMB_test_labels, Q53data.decision_function(IMB_test_features)))
    print("precision: ", performancePrecision(IMB_test_labels, Q53data.predict(IMB_test_features)))
    print("sensitivity: ", performanceSense(IMB_test_labels, Q53data.predict(IMB_test_features)))
    print("specificity: ", performanceSpec(IMB_test_labels, Q53data.predict(IMB_test_features)))
    print("f1_score: ", performanceF1(IMB_test_labels, Q53data.predict(IMB_test_features)))
    print("accuracy: ", performance(IMB_test_labels, Q53data.predict(IMB_test_features)))

    return finalAccWeights

    # print("Precision: ")
    # print(maxPrec)
    # print(finalPrecWeights)

    # print("F1: ")
    # print(maxF1)
    # print(finalF1Weights)

    # print("AUROC: ")
    # print(maxAur)
    # print(finalAurWeights)

    # print("Sensitivity: ")
    # print(maxSens)
    # print(finalSensWeights)

    # print("Specificity: ")
    # print(maxSpec)
    # print(finalSpecWeights)


    
    
def Q54(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels, negative, positive):
    Q54data = LinearSVC(penalty = "l2", loss = "hinge", C = 0.01, class_weight = {-1: 1, 1:1}, random_state=445)
    Q54data.fit(IMB_features, IMB_labels)

    x, y, thresholds = metrics.roc_curve(IMB_test_labels, Q54data.decision_function(IMB_test_features))

    #performanceAUROC(IMB_test_labels, Q54data.decision_function(IMB_test_features))

    Q54data2 = LinearSVC(penalty = "l2", loss = "hinge", C = 0.01, class_weight = {-1: negative, 1:positive}, random_state=445)
    Q54data2.fit(IMB_features, IMB_labels)

    x2, y2, thresholds2 = metrics.roc_curve(IMB_test_labels, Q54data2.decision_function(IMB_test_features))

    plt.plot(x, y)
    plt.plot(x2, y2)
    plt.legend(["neg=1, pos = 1" , "neg = 9, pos = 8"])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.show()


def Q6(features, labels, dictionary, heldout_features):

    Q6data = LinearSVC(random_state = 445, C = 1, loss= "hinge")
    Q6data.fit(features, labels)
    print("LinearSVC Accuracy with no changes:")
    print(cv_performance(Q6data, features, labels, k = 5, metric = "accuracy"))

    C_range2 = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    bestC = select_param_linear(features, labels, k = 5, C_range = C_range2, loss = "hinge")
    print("bestC: ", bestC)
    bestCdata = LinearSVC(random_state = 445, C = bestC, loss= "hinge")
    print("LinearSVC Accuracy with best C:")
    print(cv_performance(bestCdata, features, labels, k = 5, metric = "accuracy"))


    oneRest = OneVsRestClassifier(bestCdata).fit(features, labels).predict(features)
    print("One vs Rest classifie on LinearSVC with best C: ")
    print(cv_performance(bestCdata, features, labels, k = 5, metric = "accuracy"))

    Q6dataOne = LinearSVC(random_state = 445, C = bestC, loss= "hinge")
    oneOne = OneVsOneClassifier(Q6dataOne).fit(features, labels).predict(features)
    print("Linear SVC One vs One classifier with best C: ")
    print(cv_performance(Q6dataOne, features, labels, k = 5, metric = "accuracy"))

    # param_ranges = []
    # Carr2 = list(np.random.uniform(low = -2, high = 3, size = 25))
    # Rarr2 = list(np.random.uniform(low = -2, high = 3, size = 25))


    # for C_val2 in range(0, len(Carr2)):
    #     param_ranges.append((10 ** Carr2[C_val2], 10 ** Rarr2[C_val2]))


    # best = select_param_quadratic(features, labels, k = 5, metric = "accuracy", param_range= param_ranges)
    kernelOption = ["linear", "poly", "rbf", "sigmoid"]
    #kernelOption = ["linear", "poly", "rbf", "sigmoid", "precomputed"]



    # for x in kernelOption:
    #     tempSVC = SVC(C = best[0], coef0= best[1], kernel=x, degree = 2, gamma = "auto", random_state=445)
    #     tempSVC.fit(features, labels)
    #     print("Random Kernel:", x, "cv_performance: ")
    #     print(cv_performance(tempSVC, features, labels, k = 5, metric = "accuracy"))


    # Carr = [10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3]
    # Rarr = [10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3]
    # param_range = []

    # for C_val in Carr:
    #     for R_val in Rarr:
    #         param_range.append((C_val, R_val))

    # bestgrid = select_param_quadratic(features, labels, k = 5, metric = "accuracy", param_range= param_range)
    
    
    # for kern in kernelOption:
    #     tempSVC = SVC(C = bestgrid[0], coef0= bestgrid[1], kernel=kern, degree = 2, gamma = "auto", random_state=445)
    #     tempSVC.fit(features, labels)
    #     print("Grid Kernel:", kern, "cv_performance: ")
    #     print(cv_performance(tempSVC, features, labels, k = 5, metric = "accuracy"))

    # final = SVC(C = best[0], coef0=best[1], kernel= "sigmoid", gamma = "auto", degree = 2)
    # final.fit(features, labels)
    # print("Final Round Plain with c: ", best[0], "r: ", best[1])
    # print(cv_performance(final, features, labels, k = 5, metric = "accuracy"))

    # finalOne = SVC(C = best[0], coef0=best[1], kernel= "sigmoid", gamma="auto", degree=2)
    # print("Final Round OnevOne with c: ", best[0], "r: ", best[1])
    # temp = OneVsOneClassifier(finalOne).fit(features, labels).predict(features)
    # print(cv_performance(finalOne, features, labels, k = 5, metric = "accuracy"))

    C_range3 = [100, 1000]
    bestPerf = 0
    
    
    for cval in C_range3:
        finalRound = SVC(C=cval, kernel= "sigmoid", gamma="auto", degree=2)
        finalRound.fit(features,  labels)
        print("Final Round Plain with c: ", cval)
        FinalRoundPlain = cv_performance(finalRound, features, labels, k = 5, metric = "accuracy")
        print(FinalRoundPlain)

        if FinalRoundPlain > bestPerf:
            bestPerf = FinalRoundPlain
            bestclf = finalRound


        finalRoundOne = SVC(C = cval, kernel= "sigmoid", gamma="auto", degree=2)
        print("Final Round OnevOne with c: ", cval)
        temp = OneVsOneClassifier(finalRoundOne).fit(features, labels).predict(features)
        finalRoundOnevOne = cv_performance(finalRoundOne, features, labels, k = 5, metric = "accuracy")
        print(finalRoundOnevOne)

        if finalRoundOnevOne > bestPerf:
            bestPerf = finalRoundOnevOne
            bestclf = finalRoundOne

        finalRoundTwo = SVC(C = cval, kernel= "sigmoid", gamma="auto", degree=2)
        print("Final Round OnevRest with c: ", cval)
        temp = OneVsRestClassifier(finalRoundTwo).fit(features, labels).predict(features)
        finalRoundOnevRest = cv_performance(finalRoundTwo, features, labels, k = 5, metric = "accuracy")
        print(finalRoundOnevRest)

        if finalRoundOnevRest > bestPerf:
            bestPerf = finalRoundOnevRest
            bestclf = finalRoundTwo

    print("FINAL PREDICTION:")
    print(bestPerf)

    y_pred = bestclf.predict(heldout_features)
    generate_challenge_labels(y_pred, "vgbeck")





    # Q6data2 = LinearSVC(random_state = 445, C = .1, loss= "hinge")

    
    
    
    
    #Q6data2.fit(stemmedFeatures, labels)

    # print("Stemmed Performace: ")
    # print(performance(labels, Q6data2.predict(features)))



    # Q6data3 = LinearSVC(random_state = 445, C = 1, loss= "hinge")
    # lemmedFeatures = [WordNetLemmatizer.lemmatize(y) for y in stemmedFeatures]
    # Q6data3.fit(lemmedFeatures, labels)
    # print("Stemmed and Lemmed: ")
    # print(performance(labels, Q6data3.predict(features)))



    #TFIDF




    




def main():
    # Read binary data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_matrix AND extract_dictionary


    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(
        fname="data/dataset.csv"
    )

    IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = get_imbalanced_data(
        dictionary_binary, fname="data/dataset.csv"
    )

    # TODO: Questions 3, 4, 5

    # df = load_data("data/dataset.csv")
    # print(extract_dictionary(df))

    # extract_dictionary(X_train) and change line 306

    
    #Q3(X_train)
    #Q41b(X_train, Y_train)
    #Q41c(X_train, Y_train, X_test, Y_test)
    #C_range2 = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    #plot_weight(X_train, Y_train, penalty= "l2", C_range = C_range2, loss = "hinge",dual=True)
    
    #BEGINNING OF PLOT WEIGHT
    # part41e = LinearSVC(random_state = 445, C = 0.1, loss= "hinge")
    # part41e.fit(X_train, Y_train)
    # COEFF = part41e.coef_[0]
    # COEFF2 = np.argsort(COEFF)
    
    # finalIn = []

    # negative = [-1, -2, -3, -4, -5]
    # for i in negative:
    #     finalIn.append(COEFF2[i])

    # positive = [4, 3, 2, 1, 0]
    # for j in positive:
    #     finalIn.append(COEFF2[j])
    
    # finalWord = []
    # finalVal = []

    # for k in finalIn:
    #     vals = {i for i in dictionary_binary if dictionary_binary[i] == k}
    #     finalWord.append(vals)
    #     finalVal.append(part41e.coef_[0][k])
    
    # print(finalWord)
    # print(finalVal)
    
    # plt.xticks(range(len(finalWord)), finalWord, rotation = 30, ha = 'right')
    # plt.bar(range(len(finalWord)), finalVal)
    # plt.xlabel("Words")
    # plt.ylabel("Coefficient Value")
    # plt.title("Part 4.1.e")
    # plt.show()

    #END OF PLOT WEIGHT
    #higher magnitude = less slack


    #Q42(X_train, Y_train, X_test, Y_test)
    #updatedQ43(X_train, Y_train, X_test, Y_test)
    #Q43(X_train, Y_train)
    #Q51(X_train, Y_train, X_test, Y_test)
    
    #Q52(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels)
    #negpos = Q53(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels)
    #Q53(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels)
    #Q54(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels, negpos[0], negpos[1])
    
    #4.1d:
    #C_range2 = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    C_range2 = [0.001, 0.01, 0.1, 1]

    plot_weight(X_train, Y_train, penalty= "l1", C_range = C_range2, loss = "squared_hinge",dual=False)

    #plot_weight(X_train, Y_train, C_range=[0.001, 0.01, 0.1, 1, 10, 100, 1000], loss = "hinge", penalty=2, dual = True)
    #dataframe = load_data("data/dataset.csv")

    

    # Read multiclass data
    # TODO: Question 6: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels

    
    (multiclass_features,
    multiclass_labels,
    multiclass_dictionary) = get_multiclass_training_data()
    
    heldout_features = get_heldout_reviews(multiclass_dictionary)

    Q6(multiclass_features, multiclass_labels, multiclass_dictionary, heldout_features)


if __name__ == "__main__":
    main()
    
