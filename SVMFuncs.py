'''
----Kernel Selection & Stylometry Classification Functions----
This document contains the block of functions required to repeat the experiment in the 'Kernel Selection...' paper.

The functions herein come from an early attempt (2022) at creating stylometry tools -- and many have since been improved.
While this code should accurately help you re-create and reproduce the experiments in the paper, some of the other arguments included here (such as niche KNN distances) may not be functional.
If you intend to use this code for anything other than this explicit experiment, I would encourage you to contact me instead (at nathan.dooner at gmail dot com) -- and I'll happily provide whatever I can to support you. 


Functions immediately below this comment are for generating a matrix of word frequency counts. 
Brief comments are included below other blocks, explaining what they pretain to. 
'''

def DFM(textSet, MFW = 100, Culling = 0, FactorANON = True, preserveCaps = False, Counts = "relative",
        Ceiling = 0, removePronouns = False, removeSelection = [], preservePunctuation = [], ChunkVol = 0, ChunkType = "count",
        Export = None):
    from collections import Counter; import pandas as pd; from os.path import isdir; from pathlib import Path; import string
    dfList = [] #Later used to store text file Word Freqs
    
    if Counts == "raw": rawCounts = [] 
    elif Counts != "relative": 
        print("Error in Count Argument; setting to relative counts")
        Counts = "relative" 
    
    #--textSet Data Preparation--
    #"textSet" takes either a List of text FilePaths, or a File Path containing a folder of texts.
    #Check if it's a list of file paths; if so, just add a "Corpus" pointer. 
    if (ChunkVol > 0) and (ChunkType == "count"):
        path = Splitter(textSet, n = ChunkVol, method = ChunkType)
        Corpus = Data_Prep(path, full = True)
        TextNames = Data_Prep(path, full = False)
        
    elif (ChunkVol > 0) and (ChunkType == "percent"):
        path = Splitter(textSet, n = ChunkVol, method = ChunkType)
        Corpus = Data_Prep(path, full = True)
        TextNames = Data_Prep(path, full = False)
        
    elif type(textSet) == list:
        if Path(textSet[0]).is_file():
            Corpus = textSet
            TextNames = [entry.split("/")[-1] for entry in Corpus]
            
    #Else treat it as a single file path, and pull all texts inside
    elif isdir(textSet):
        Corpus = Data_Prep(textSet, full = True)
        TextNames = Data_Prep(textSet, full = False)
        
    for i in range (0, len(Corpus)): 
        with open(Corpus[i], "r",) as Document:
            Document = Document.read()
                
            if preserveCaps == False:
                Document = Document.lower()
            
            punctuationToRemove = string.punctuation
            if len(punctuationToRemove) > 0:
                punctuationToRemove = [entry for entry in punctuationToRemove if entry not in preservePunctuation]
                punctuationToRemove = "".join(punctuationToRemove) #Translate takes string. This makes it right format. 
                
            
            Document = Document.translate(str.maketrans(' ', ' ', punctuationToRemove)) #Strips punctuation
            Document = Document.split()
    
            #Create a dictionary of Raw Word Counts across a text
            Doc_Stats = dict(Counter(Document))
        
            #Get the total number of Words in a text
            Word_Count = len(Document)
            
            #Divide the number of occurences of each word by the total number of words, creating a dictionary.
            Relative_Dict = {Key:((Doc_Stats[Key]/Word_Count) * 100) for Key in Doc_Stats}
            
            #Add the dictionary to temporary list
            dfList.append(Relative_Dict)               

            if Counts == "raw":
                rawCounts.append(Doc_Stats)                                                
    
    #Turn the list of dictionaries into a Dataframe
    Combined_DF = pd.DataFrame(dfList, index = TextNames) 
    
    #If Raw Counts are being considered, create a Raw_DF so we can extract a Sorter() from that
    if Counts == "raw":
        Raw_DF = pd.DataFrame(rawCounts, index = TextNames)
    
    if removePronouns == True or len(removeSelection) > 0:
        Combined_DF = DropFromDFM(Combined_DF, pronouns = removePronouns, Selection = removeSelection)
    
        
    #Sorter Value holds array of column sums, and sorts DF in descending order of size. We can include/exclude ANON plays for tallying these column sums.
    if (Ceiling > 0) and Counts == "relative":
        Combined_DF = CeilingSort(Matrix = Combined_DF, IncludeANON = FactorANON, ceiling = Ceiling)
        
    else:
        if (FactorANON == False):             
             Anon_Excluded = [entry for entry in Combined_DF.index if "ANON_" not in entry]
             if Counts == "raw":
                 No_Anon_Set = Raw_DF.loc[Anon_Excluded] 
             else: #Counts == "relative'
                 No_Anon_Set = Combined_DF.loc[Anon_Excluded]
             Sorter = No_Anon_Set.sum()    
        else:
            if Counts == "raw":
                Sorter = Raw_DF.sum()
            else: #Counts == "relative"
                Sorter = Combined_DF.sum()
            
        #We use Combined_DF regardless of sorting method, because we obviously don't want just the raw counts, even if we're going off them.    
        Combined_DF = Combined_DF[Sorter.sort_values(ascending = False).index]
    
    if Culling > 0:
        CullingThresh = int ( len(Combined_DF.index) * (Culling/100) ) 
        Combined_DF.dropna(axis = 1, thresh = CullingThresh, inplace = True)
        
    WordMatrix = Combined_DF.iloc[0:(len(Combined_DF.index)), 0:(MFW)] 
    WordMatrix.fillna(0, inplace = True)
    
    if type(Export) == str:
        WordMatrix.to_csv(Export)
    
    return (WordMatrix)

def CeilingSort(Matrix, IncludeANON, ceiling = 3.5):
    SumList = []
    for i in range (0, len(Matrix.columns)):
        Column = Matrix[Matrix.columns[i]]
        Sum = 0
        for j in range (0, len(Column)):
            if ( (IncludeANON == True) or ("ANON_" not in Column.index[j]) ):         
                if Column[j] > ceiling:
                    Sum += ceiling
                else: 
                    Sum += Column[j]
        SumList.append(Sum)
    
    Matrix.loc['Averages'] = SumList #Appending list as row to DF
    Matrix.sort_values(by = 'Averages', axis = 1, ascending = False, inplace = True)
    Matrix.drop('Averages', inplace = True)
        
    return(Matrix)


def DropFromDFM(WordDFM, pronouns = False, Selection = []):
    pronounsToPurge = ["i", "we", "you", "thou", "he", "she", "it", "they", "me", "us", "thee", "him", "her", "them", "our", "their"]
    
    if pronouns == True:
        Selection.extend(pronounsToPurge)
    
    ToPurge = [entry for entry in Selection if entry in WordDFM]
    New_DFM = WordDFM.drop(labels = ToPurge, axis = 1)
    return New_DFM

def Splitter(inp, n, method):    
    from math import floor; import os; import shutil; from pathlib import Path
    
    if type(inp) == list:
        if Path(inp[0]).is_file():
            x = inp
            y = [entry.split("/")[-1] for entry in x]
            out_path = x[0].replace("/" + y[0], "") + "/Chunks/"
    else: 
        x = Data_Prep(inp, full = True) 
        y = Data_Prep(inp, full = False) #Y is used within the 'try' bit for creating the export file path
        out_path = inp + '/Chunks/'

    
    if os.path.exists(out_path) == True:
        shutil.rmtree(out_path)
        os.makedirs(out_path)
    else: 
         os.makedirs(out_path)
         
    if method == "percent":
        Steps = int(100 / n) #Get the number of iterations for the for loop
        
        for j in range (0, len(x)): #Iteratre Through Texts
            text = open(x[j]).read().split() #Open a text
            Portion = int (len(text) * (n/100)) #Get the desired word count for each step of text
            
            for k in range (0, Steps): #For however many full chunks you can make of that text
                if k+1 == Steps:
                    segment = ' '.join(text[Portion * k:]) #Take a segment of that text
                else:
                    segment = ' '.join(text[Portion * k: Portion * (k+1)]) #Take a segment of that text
                file_path = out_path + y[j] + '_' + str(k)
                if (".txt" in file_path):
                    file_path = file_path.replace(".txt", "")
                file_path += ".txt"
                
                output = open(file_path, 'w') #Open it as a file path
                output.write(segment) #Write the segment
                
    
    else: # if method == count.        
        for j in range (0, len(x)): #Iteratre Through Texts
            text = open(x[j]).read().split() #Open a text
            for k in range (0, int((len(text)/n))): #For however many full chunks you can make of that text
                segment = ' '.join(text[n * k: n * (k+1)]) #Take a segment of that text
                file_path = out_path + y[j] + '_' + str(k)
                if (".txt" in file_path):
                    file_path = file_path.replace(".txt", "")
                file_path += ".txt"
                
                output = open(file_path, 'w') #Open it as a file path
                output.write(segment) #Write the segment
                

    return(out_path)


def Data_Prep (data, full=False):
    from os import listdir
    Corpus_List = listdir(data)
        
    for item in Corpus_List:
        if item.endswith(".txt") == False:
            Corpus_List.remove(item)
          
    if full == True:
        for i in range (0, len(Corpus_List)):
            Corpus_List[i] = data + '/' + Corpus_List[i]
        return(Corpus_List)
    else: return (Corpus_List)

def splitter(inp, n=2):    
    from math import floor; import os; import shutil
    
    x = Data_Prep(inp, full = True) 
    y = Data_Prep(inp, full = False) #Y is used within the 'try' bit for creating the export file path

    out_path = inp + '/Chunks/'
    
    
    if os.path.exists(out_path) == True:
        shutil.rmtree(out_path)
        os.makedirs(out_path)
    else: 
         os.makedirs(out_path)

    for j in range (0, len(x)):
        text = open(x[j]).read()    
        step = len(text)/n 
        i = 0
        while i < n:
                try: 
                    output = open(out_path + y[j] + '_' + str(i) + '.txt', 'w')
                    output.write( text [floor((step*i)) : floor((step*(i+1))) ])
                    
                    i += 1
                except:
                    print("Error in chunking! At " + str(y[j]))
                    i += 1
                    continue  
                

    return(out_path)


'''
KNN  Classification
'''

def KNN_Classify(TrainingData, TrainingClasses, TestData, TestTitles = None, Distance = "manhattan",
                 Neighbours = 1, Weights = "uniform", Algorithm = "auto", LeafSize = 30, P = 2, 
                 Metric = "minkowski", MetricParams = None, N_jobs = None, Radius = None):
    
    from sklearn.neighbors import KNeighborsClassifier
    if (Distance.lower() == "euclidean"):
        P = 2
    elif (Distance.lower() == "manhattan"):
        P = 1
    elif (Distance.lower() == "cosine"):
        Metric = "cosine"
    elif (Distance.lower() == "chebyshev"):
        import math
        infinity = math.inf
        P = infinity
    elif (Distance.lower() == "mahalanobis"):
        '''
        Requires a covariance matrix is passed, effectively a MFW Dataframe of both the test and training data together, which Pandas can convert to 
        a covariance matrix with the DF.cov() function. 
        
        Apparently excellent explanation of method: https://stats.stackexchange.com/questions/62092/bottom-to-top-explanation-of-the-mahalanobis-distance/62147
        
        '''
        import pandas as pd
        Metric = "mahalanobis"
        Algorithm = "brute"
        MFW = pd.concat([TrainingData, TestData])
        MetricParams={"VI": MFW.cov()}


    try:
        knn = KNeighborsClassifier(n_neighbors=Neighbours, weights = Weights, algorithm = Algorithm, leaf_size = LeafSize, p = P, 
                               metric = Metric, metric_params = MetricParams, n_jobs = N_jobs, radius = Radius)
        
        knn.fit(TrainingData, TrainingClasses)
        prediction = knn.predict(TestData)
    except ValueError:
        #print("Value Error! This is likely due to too samples to satisfy k value. Occurred at k value " + str(Neighbours))
        #print("Appending Results as ERROR and continuing with classification")
        NeighbourVal = str(Neighbours)
        Results = [("Value Error at K Value:,", NeighbourVal)]
        return Results
    
    Results = [ str(entry) for entry in prediction ]
    if TestTitles != None:
        if len(Results) == 1: 
            Results = Results[0]
            Results = (TestTitles, Results)
        else:
            Results = list(zip(TestTitles, Results))
    
    return (Results)
                         
'''
    ***** Distance Measures *****
    
    https://journals.sagepub.com/doi/epdf/10.1177/1176935120965542 Very good source + Writeup of applying KNN distance measures in med
    Manhattan distance is Minkowski when p = 1, 
    Eculidean distance is Minkwoski when p = 2,
    Chebyshev distance is Minkowski when p = [infinity],
    '''
    
'''
SVM classification

'''
def SVM_Classify(TrainingData, TrainingClasses, TestData, TestTitles = None, Kernel = "rbf", C = 1.0, Degree = 3.0, Gamma = "scale", paramReturn = False):
    #Nate Note: TestTitles not used, but kept here for potential later validation functionality.
    from sklearn import svm;
    clf = svm.SVC(kernel=Kernel, decision_function_shape = "ovo", C = C, degree = Degree, gamma = Gamma) 
    clf.fit(TrainingData, TrainingClasses) #Train the model using the training sets
    prediction = clf.predict(TestData)
    
    SVM_Output = [str(entry) for entry in prediction]
    if TestTitles != None:
        SVM_Output = [(TestTitles[i], SVM_Output[i]) for i in range (0, len(TestTitles))]
    
    if (paramReturn == True):
        parameters = clf.get_params()
        return SVM_Output, parameters
    else:
        return SVM_Output



'''

Incidence & Evidence Functions -- 'flatten' used for modifying the lists of dataframes / tests, so they can be better iterated through
'UpdateDict' called as a one-liner in various loops. to either Update a particular key, or generate that key if it doesn't exist already
'''
def flatten(xss):
    return [x for xs in xss for x in xs]

def UpdateDict(dictionary, key):
    if key in dictionary:
        dictionary[key] += 1
    else:
        dictionary[key] = 1
        
        
'''
Cartesian Product -- Given a matrix of word freqs ('DFM'), create a set of 1v1 experiments for later iterating through. 
'''
#Given a Matrix of Word Freqs, and a number *N*, return a list of 4 lists, containing every possible combination of authors using *N* texts from each author in every combination, with
    #List 1 - The indexes of each texts, to allow for DFM.iloc extraction
    #List 2 - Tuples containing (AUTHOR_TEXT_TITLE_1, AUTHOR_TEXT_TITLE_2, etc.) for each combination
    #List 3 - The iloc of All of the Anonymous Texts within the DFM
    #List 4 - Tuples containing (ANON_TEXT_TITLE) for each Anonymous Text within the DFM

#'Retain Splits' boolean ensures that, if texts are broken into chunks, the ilocs etc. added to each list considers every chunk of the same text as one whole
#!!! Note -- The function won't work if RetainSplits = True and there are no splits. This isn't a bug, but it's something you have to keep in mind.


def CartesianProduct(DFM_To_Use, OnlyDiffAuthors = True, RetainSplits = False, TPA = 2):
    from itertools import combinations; from itertools import product; from itertools import chain
    ComboSet = [ [], [], [], [] ] #List of 4 Lists, which will only data that is returned at end.
    Original = list(DFM_To_Use.index) #List of indexes without removing chunk numbers. Later used to refer to chunk titles and indexes

    if RetainSplits == True:
        Texts = list(DFM_To_Use.index)
        Texts = [entry.rsplit("_",1)[0] for entry in Texts] #Split at last delimiter, and throwaway end bit
        Texts = list(set(Texts)) #Set function retains only unique elements, list function re-casts the set as a list again.
        Texts = [entry.split("_", 1) for entry in Texts]
               
    else:
        #Below Line Assumes a DFM has been fed into the function
        Texts = [entry.split("_", 1) for entry in list(DFM_To_Use.index)]
    
    ANON_Texts = [entry for entry in Texts if "ANON" in entry[0]]
    Training_Texts = [entry for entry in Texts if "ANON" not in entry[0]] 
    
    
    #Iterate through Authors in Training_Texts, creating a list of lists, where every list corresponds to its own author's training data
    Training_Lists = []
    Author_Set = list(set([entry[0] for entry in Training_Texts])) #List of Unique Author Names from Training Data
    for author in Author_Set:
        AuthorWorks = ['_'.join(entry) for entry in Training_Texts if author in entry]
        Training_Lists.append(AuthorWorks)
        

    #Turn List of Lists of Authors into a list of every author's own training data combinations
    Author_Combinations = [list(combinations(author, TPA)) for author in Training_Lists] 
    #...then create every combination of every list of combinations
    Combinations = list(product(*Author_Combinations))
    
    
    ANON_Indices = [i for i, s in enumerate(Original) if "ANON" in s] #Pull all the indices from the Chunks of text that correspond to Anonymous works
    ANON_Titles = [Original[i] for i in ANON_Indices] #Pull all the titles from the Chunks of text that correspond to Anonymous works
        
    
    for entry in Combinations:
        TD_Indexes, TD_Titles = [], []
        #Turn list of lists of texts into just a list of texts
        Training_Data = list(chain.from_iterable(entry))
        
        for text in Training_Data:    
            #Author Data Index for DFM iloc, taking chunk indices from list 'Original'
            Author_Indexes = [i for i, s in enumerate(Original) if text in s]
            #Author Data Titles taking chunk designations from list 'Original' 
            Author_Titles = [Original[i] for i in Author_Indexes]

            TD_Indexes.append(Author_Indexes)
            TD_Titles.append(Author_Titles)
        
        TD_Indexes = list(chain.from_iterable(TD_Indexes))
        TD_Titles = list(chain.from_iterable(TD_Titles))
        
        #Training Data Index for DFM iloc
        ComboSet[0].append(TD_Indexes)
        #Training Data Title
        ComboSet[1].append(TD_Titles)        
        #Text treated as Anon for DFM iloc
        ComboSet[2].append(ANON_Indices)        
        #Text treated as Anon's title
        ComboSet[3].append(ANON_Titles)


    return ComboSet

    
        