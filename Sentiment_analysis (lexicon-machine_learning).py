import pandas as pd
import numpy as np
import csv
from IPython.display import display
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
import re


"""

This module is used for creating sentiment analysis training and test datasets from a sentiment analysis lexicon. 
It is also used to get performance metrics about machine learning classifiers. 
It uses the training dataset to train the model and tests on the test dataset.

"""

class Create_sentiment_Dataframe:
    
    """
    
    Create_sentiment_Dataframe uses a lexicon file containing word, sentiment score combos and a file containing blocks of text.
    When called it creates a dataframe, with each instance being a block of text, features = lexicon words and frequency they appear in block of text.
    
    """
    
    def __new__(cls, lexicon, text):
        
        """
        
       Construct a new 'Create_sentiment_Dataframe' object.

       :param lexicon: A file containing words and corresponding sentiment score, tab separated
       :param text: A file containing blocks of text
       :return: Returns a populated dataframe
       
       """
        
        lexicon_dic = cls.create_lexicon_dic(lexicon)
        words, labels = cls.get_words_from_reviews(text)
        word_frequency_lists = cls.get_lists_word_frequency(words, cls.get_word_frequency, lexicon_dic)
        dataframe = cls.add_freq_to_dataframe(word_frequency_lists, lexicon_dic, labels)
        
        return dataframe
    
        
    def create_lexicon_dic(lexicon):
        
        """
        
       Creates a dictionary. Keys = words, values = sentiment scores

       :param lexicon: A file containing words and corresponding sentiment score, tab separated
       :return: Returns dictionary object.
       
       """
        
        lexicon_dic = dict()
        with open(lexicon) as file:
            tsv_file = csv.reader(file, delimiter="\t")
            
            for line in tsv_file:
                lexicon_dic[line[0]] = line[1]
                
        
        return lexicon_dic
           
    
    def get_words_from_reviews(text):
        
         
        """
        
       Gets all the words from the text blocks.

       :param text: A file containing blocks of text
       :return: A list containing words
       
       """
        
        text_sentences = []
        text_words = []
        
        with open(text) as file:
            tsv_file = csv.reader(file, delimiter="\n")
            
            for line in tsv_file:
                text_sentences.append(line[0])
                
        for i in range(len(text_sentences)):
            text_words.append(text_sentences[i].split(" "))
        
        
        labels = []
        
        for i in text_words:
            i = re.sub("[^0-9]", "", i[len(i) - 1])
            labels.append(i[len(i) - 1])
 

        return text_words, labels
    
   
    def get_word_frequency(string, lexicon_dic):
        
        """
        
       Calculates frequency that lexicon words appear in blocks of text.

       :param string: A list of words
       :param lexicon_dic: A dictionary containing sentiment words and scores.
       :return: Returns a list containing frequence lexicon words appeared in block of text, in order.
       
        """
       
        from collections import Counter
        
        words = lexicon_dic.keys()
    
        word_counts = Counter(string)
        
        word_and_freq_list = []
        count = []
        
        for word in words:
          word_and_freq = word, word_counts[word]
          word_and_freq_list.append(word_and_freq)
    
        for i in word_and_freq_list:
            count.append(i[1])
        
        
        return count
      
        
    def get_lists_word_frequency(words, get_word_frequency, lexicon_dic):
    
         """
            
         Uses the get_word_frequency function to get a 2D list containing lexicon word frequencys for all blocks of text.
    
         :param words: A list containing all words from file containing blocks of text
         :param get_word_frequency: A function to count the word frequencys
         :param lexicon_dic: A dictionary containing words and sentiment scores.
         :return: Returns a 2D Numpy array containing frequency of words appeared in text blocks. Every index is a full run of lexicon words and frequency appeared in each text block
           
        """
         
         word_freq_sub_lists = [[]]
         
         for i in words:
             word_freq_sub_lists += (get_word_frequency (i, lexicon_dic))
     
         del word_freq_sub_lists[0]
         word_freq_sub_lists = np.array(word_freq_sub_lists, dtype=object)
         word_freq_sub_lists = np.split(word_freq_sub_lists, 5000)
        
        
         return word_freq_sub_lists
   
    
    def add_freq_to_dataframe(wordfreq, lexicon_dic, labels):
        
            """
           
            Converts the frequencys words appeared in the blocks of text to their corresponding sentiment scores and creates a dataframe with them.
            dataframe features are the lexicon words.
           
            :param wordfreq: A 2D numpy array containing frequencys of lexicon words.
            :param lexicon_dic: A dictionary containing words and corresponding sentiment scores.
            :return: A populated dataframe
          
            """
           
            
            headings = []
            
            for i in lexicon_dic.keys():
                headings.append(i)
            headings.append("Positive/Negative")
            
            lexi_vals = list(lexicon_dic.values())
            
           
            for listind, j in enumerate(wordfreq):
                    for elementind, i in enumerate(j):
                        if (i > 0):
                            wordfreq[listind][elementind] = i * float(lexi_vals[elementind])
            
            
            count = 0
            ind = 0
            for listind, j in enumerate(wordfreq):
                        
                        for i in j:
                            count += i
                            ind += 1
                                
                            if ind == len(wordfreq[listind]):
                                
                                if (count > 1):
                                    count = 1
                                elif count < 1:
                                    count = 0
                                    
                                wordfreq[listind] = np.append(wordfreq[listind], count)
                                count = 0
                                ind = 0
                        
           
            dataframe = pd.DataFrame(wordfreq, columns=[headings])
          
            return dataframe
        


lexicon_file_path = "C:\\Users\\kntjk\\Desktop\\uni\\One_drive_temp\\Intelligent_systems\\CW2\\Datasets\\advanced\\Lexicon file"
training_data_path = "C:\\Users\\kntjk\\Desktop\\uni\\One_drive_temp\\Intelligent_systems\\CW2\\Datasets\\advanced\\raw_text_data\\reviews_Video_Games_training.raw.tsv"
test_data_path = "C:\\Users\\kntjk\\Desktop\\uni\\One_drive_temp\\Intelligent_systems\\CW2\\Datasets\\advanced\\raw_text_data\\reviews_Video_Games_test.raw.tsv"


training_dataframe = Create_sentiment_Dataframe(lexicon_file_path, training_data_path)

test_dataframe = Create_sentiment_Dataframe(lexicon_file_path, test_data_path)



class run_classifier:
    
    """
          
    Class uses a training and test dataframe to print he performance metrics of an estimator containing machine learning classifiers.
    
    """

    def __init__(self, training_dataframe, test_dataframe, estimators, model): 
        
        """
        
       Construct a new 'run_classifier' object.

       :param training_dataframe: A populated dataframe to train on
       :param test_dataframe: A populated pandas dataframe to test on
       :param estimators: An estimator containing machine learning classifiers
       
       """

        self.confidence_vote_classifier(training_dataframe, test_dataframe, estimators, self.get_dataframe_names_instances_labels, model)
    
#        self.predict_classifier(training_dataframe, self.get_dataframe_names_instances_labels)


    def get_dataframe_names_instances_labels(self, dataframe):
        
        """
        
       Gets the feature names, instances and labels from a dataframe.

       :param dataframe: A popualted pandas dataframe
       :param estimators: An estimator containing machine learning classifiers
       :return: Returns feature names, instances and labels
       
       """
       
        frame = dataframe
        headings = list (frame.columns.values)
        feature_names = headings[:len(headings) - 1]
        label_name = headings[len(headings) - 1: len (headings)] [0]
        frame = frame._get_numeric_data()
        numpy_array = frame.to_numpy()
        number_rows, number_columns = numpy_array.shape
        instances = numpy_array [:, 0: number_columns - 1]
        labels = []
        print(number_columns)
        for label in numpy_array [:, number_columns - 1 : number_columns].tolist():
             labels.append(label[0])
        labelss = np.asarray(labels, dtype=np.float32)
         
          
        return feature_names, instances, labels
    
 
    def predict_classifier (self, dataframe, get_dataframe_names_instances_labels):
        
         """
        
       Displays top performing sklearn classifiers on a dataframe containing a set of features.

       :param dataframe: A popualted pandas dataframe
       :param get_dataframe_names_instances_labels: A function that gets the feature names, instances and labels of a dataframe
       
         """
        
         from sklearn.model_selection import train_test_split
         from lazypredict.Supervised import LazyClassifier
        
         training_names, training_instances, training_labels = get_dataframe_names_instances_labels(dataframe)
    
         X = training_instances
         y = training_labels
         
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5, random_state = 123)
    
         clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
         models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    
         display(models.to_string())
        
    
    def confidence_vote_classifier (self, training_csv, test_csv, estimators, get_dataframe_names_instances_labels, model):
        
        get_dataframe_names_instances_labels(training_dataframe)  
        
        
        """
      
        Uses the confidence vote to predict a target binary labels on data.

        :param training_csv: A dataframe containing training data
        :param test_csv: A dataframe containing test data
        :param estimators: A list of estimators containing machine learning classifiers
        :param get_dataframe_names_instances_labels: A function that gets the feature names, instances and labels of a dataframe
        
        """
        
        from sklearn.metrics import classification_report
        from sklearn.ensemble import VotingClassifier
        from sklearn.model_selection import KFold, cross_val_score
        from sklearn.model_selection import RepeatedKFold
        from sklearn.model_selection import RandomizedSearchCV
        
        
        
#        lgbm_param = {
#        "num_leaves": [20, 40, 60, 80, 100],
#        "min_data_in_leaf": [2, 4, 5, 6, 7, 10, 20, 30, 40, 50],
#        "feature_fraction": [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8],
#        "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#        "num_iterations": [100, 200, 300, 350, 400, 500, 600, 700],
#        "max_depth": [10, 30, 50, 70, 90, 100, 130, 150, 180, 200],
        
#        }
        

        training_names, training_instances, training_labels = get_dataframe_names_instances_labels(training_csv)
        test_names, test_instances, test_labels = get_dataframe_names_instances_labels(test_csv)
        
        majority_vote_classifier = VotingClassifier(estimators = estimators,
                                                    voting = "soft")
        
        majority_vote_classifier.fit(training_instances, training_labels)
    
        predicted_labels = majority_vote_classifier.predict(test_instances)
        

#        k_folds = RepeatedKFold (n_splits = 10, n_repeats = 3, random_state = 5)

        
#        random_search = RandomizedSearchCV(estimator = model, param_distributions = lgbm_param, cv = k_folds, n_iter = 100, verbose = 1, scoring = "f1", n_jobs = -1)
        
#        result = random_search.fit(training_instances, training_labels)
        
#        print(result.best_score_)
#        print(result.best_params_)

        
        print(classification_report(test_labels, predicted_labels, digits = 3))
       

    
ada = AdaBoostClassifier(n_estimators = 300, random_state = 5)
knn = KNeighborsClassifier(n_neighbors = 50)
lgbm = LGBMClassifier (random_state = 5, num_leaves = 20, num_iterations = 500, min_data_in_leaf = 30, max_depth = 150, learning_rate = 0.3, feature_fraction = 0.2)

estimators = [("class1", lgbm)]
              #("class2", log_reg),
              #("class3", tree)]
    
    
run_classifier(training_dataframe, test_dataframe, estimators, lgbm)  
