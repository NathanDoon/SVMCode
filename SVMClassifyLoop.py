''' Replication '''
from os import listdir
Source = "/Users/nate/Documents/Academia/DMU PhD Year 2/SVM Study/Materials/Texts"
FolderSet = [(Source + "/" + entry + "/Full Corpus") for entry in listdir(Source) if ".DS_Store" not in entry]

ChunkSet = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000]   # Covered
MFWSet = [100, 150, 200, 250, 300, 350, 400, 450, 500]                  # Covered
CSet = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]                      # Covered
SVM_Ker =["linear", "rbf", "poly"]                                              # Covered
Poly_Degree = [2.0, 3.0, 4.0, 5.0, 6.0]                                         # Covered

Results = []

for folder in FolderSet:
    for Chunk_Val in ChunkSet:
        WordMatrix = DFM(folder, MFW = MFWSet[-1], ChunkVol = Chunk_Val)
        for MFW_Val in MFWSet:
            MatrixSlice = WordMatrix.iloc[:MFW_Val].copy()
            
            RowNames = MatrixSlice.index #Used to Pull Classes
            Train_Classes = [ entry.split("_")[0] for entry in RowNames if "ANON" not in entry] #Take just authors
            if (len(set(Train_Classes)) != 5): #This ensures our MFW/Chunk combination is valid for all authors. If not, skip. 
                continue
            classes_to_pull = [ entry for entry in RowNames if "ANON" not in entry]
            Train_Data = MatrixSlice.loc[classes_to_pull] # Word Freq's 
            Test_Classes = [ entry for entry in RowNames if "ANON" in entry ] #Take just authors
            Test_Data = MatrixSlice.loc[Test_Classes] # Take ANON DFM Row(s)
            
            for Kernel_Val in SVM_Ker:
                for C_Val in CSet:
                    if Kernel_Val == "poly":
                        for Poly_Val in Poly_Degree:
                             Outcome = SVM_Classify(Train_Data, Train_Classes, Test_Data, Test_Classes, Kernel = Kernel_Val, 
                                           C = C_Val, Degree = Poly_Val, Gamma = "scale", paramReturn = False)
                    else:
                        Outcome = SVM_Classify(Train_Data, Train_Classes, Test_Data, Test_Classes, Kernel = Kernel_Val, 
                                               C = C_Val, Degree = 3.0, Gamma = "scale", paramReturn = False)
                    
                    for entry in Outcome:
                        ToAppend_Text = entry[0]
                        ToAppend_Author = entry[1]
                        ToAppend_MFW = MFW_Val
                        ToAppend_Chunk = Chunk_Val
                        ToAppend_Kernel = Kernel_Val
                        ToAppend_C = C_Val
                        if Kernel_Val == "poly":
                            ToAppend_Poly = Poly_Val
                        else:
                            ToAppend_Poly = 0
                        
                        ToAppend_Tuple = (ToAppend_Text, ToAppend_Author, ToAppend_MFW, 
                                          ToAppend_Chunk, ToAppend_Kernel, ToAppend_C, ToAppend_Poly)
                        
                        Results.append(ToAppend_Tuple)


import pandas as pd
x = pd.DataFrame(data = Results, columns = ["Text", "Author", "MFW", "Chunk", "Kernel", "C", "Poly"], )
x.to_csv("/Users/nate/Desktop/SVMData.csv")

    
''' End of New Replication Block '''

from os import listdir
Source = "/Users/nate/Documents/Academia/DMU PhD Year 2/SVM Study/Materials/Texts"
FolderSet = [(Source + "/" + entry + "/Full Corpus") for entry in listdir(Source) if ".DS_Store" not in entry]
FolderRez = []

#Run 0
for folder in FolderSet:
    x = Ranged_Classify(folder, MFW = [10, 50], Culling = [0], 
                        Chunking = [1000, 1500, 2000], ChunkType = "count",
                        classifiers = ["svm"], SVM_Ker =["linear", "rbf", "poly"], C = [10.0, 15.0, 20.0, 25.0, 30.0], 
                        SVM_Degree = [2.0, 3.0, 4.0, 5.0, 6.0], paramReturn = True, Norm = "zscore")
    FolderRez.append(x)

#Run 1
for folder in FolderSet:
    x = Ranged_Classify(folder, MFW = [100, 150, 200], Culling = [0], 
                        Chunking = [1000, 1500, 2000], ChunkType = "count",
                        classifiers = ["svm"], SVM_Ker =["linear", "rbf", "poly"], C = [10.0, 15.0, 20.0, 25.0, 30.0], 
                        SVM_Degree = [2.0, 3.0, 4.0, 5.0, 6.0], paramReturn = True, Norm = "zscore")
    FolderRez.append(x)

#Run 2
for folder in FolderSet:
    x = Ranged_Classify(folder, MFW = [100], Culling = [0], 
                        Chunking = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000], ChunkType = "count",
                        classifiers = ["svm"], SVM_Ker =["linear", "rbf", "poly"], C = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 
                        SVM_Degree = [2.0, 3.0, 4.0, 5.0, 6.0], paramReturn = True, Norm = "zscore")
    FolderRez.append(x)
    
    
#Run 3
for folder in FolderSet:
    x = Ranged_Classify(folder, MFW = [200], Culling = [0], 
                        Chunking = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000], ChunkType = "count",
                        classifiers = ["svm"], SVM_Ker =["linear", "rbf", "poly"], C = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 
                        SVM_Degree = [2.0, 3.0, 4.0, 5.0, 6.0], paramReturn = True, Norm = "zscore")
    FolderRez.append(x)
 
#Run 4
for folder in FolderSet:
    x = Ranged_Classify(folder, MFW = [300], Culling = [0], 
                        Chunking = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000], ChunkType = "count",
                        classifiers = ["svm"], SVM_Ker =["linear", "rbf", "poly"], C = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 
                        SVM_Degree = [2.0, 3.0, 4.0, 5.0, 6.0], paramReturn = True, Norm = "zscore")
    FolderRez.append(x)

#Run 5
for folder in FolderSet:
    x = Ranged_Classify(folder, MFW = [400], Culling = [0], 
                        Chunking = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000], ChunkType = "count",
                        classifiers = ["svm"], SVM_Ker =["linear", "rbf", "poly"], C = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 
                        SVM_Degree = [2.0, 3.0, 4.0, 5.0, 6.0], paramReturn = True, Norm = "zscore")
    FolderRez.append(x)

#Run 6
for folder in FolderSet:
    x = Ranged_Classify(folder, MFW = [500], Culling = [0], 
                        Chunking = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000], ChunkType = "count",
                        classifiers = ["svm"], SVM_Ker =["linear", "rbf", "poly"], C = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 
                        SVM_Degree = [2.0, 3.0, 4.0, 5.0, 6.0], paramReturn = True, Norm = "zscore")
    FolderRez.append(x)    


#ToOutput line doesn't work with SVM    
ToOutput = [item for sublist in Results for item in sublist]


def Ranged_Classify(
             Path, MFW = [], Culling = [], Chunking = [], ChunkType = [], 
             classifiers = "Delta",
             DelDist = "Manhattan", ByAuthor = True, FactorAnon = True,
             KnnDist = "Manhattan", Knn_K = 3, 
             SVM_Ker = "rbf", C = 1.0, SVM_Degree = 3.0, SVM_Gamma = "auto", paramReturn = False, Norm = None):
    import pandas as pd;  from scipy.stats import zscore; from scipy.cluster.vq import whiten


#######PERCENTAGE TEST END

#NATE NOTE: Changed KNN Try/Except to include commas in the error statement if not enough near neighbours. Very temporary fix to allow results maker to work, as it splits on commas,
#...and otherwise wouldn't be able to parse entries for DFM without them in the results column for the KNN Errors. 

import pandas as pd; from collections import Counter
flat_list = [item for sublist in ToOutput for item in sublist]
To_Parse = []
for i in range (0, len(flat_list)):
    for j in range (0, len(flat_list[i][0])):
        Classification = str(flat_list[i][0][j]) #Text / Classification Tuple
        Classification = Classification.replace("'", "")
        Classification = Classification.replace("(", "")
        Classification = Classification.replace(")", "")
            
        Values = str(flat_list[i][1])
        Values = Values.replace("'", "")
        Values = Values.replace("(", "")
        Values = Values.replace(")", "")
            
        ToAppend = (Classification, Values)
        To_Parse.append(ToAppend)
            
DF = pd.DataFrame(data = ToOutput, columns = ["Classifications", "Values"])
    
#Find columns containing Value Error String
error_rows = DF[DF['Classifications'].str.contains('Value Error')].index
#Drop Those Columns
DF.drop(index = error_rows, inplace = True)    

DF[["Text", "Classification"]] = DF.Classifications.str.split(", ", expand = True)
    
DF[["Classifier", "Kernel/Dist", "MFW","Culling", "Chunking", "ChunkVol", "Paramter 1", "Parameter 2", "Parameter 3"]] = DF.Values.str.split(", ", expand = True)
#Need to be checking the number of parameters before splitting. Above line is needed for Delta / SVM but KNN uses below line.
#DF[["Classifier", "Kernel/Dist", "MFW","Culling", "Chunking", "ChunkVol", "Paramter 1"]] = DF.Values.str.split(", ", expand = True)
    
DF.drop(labels = ["Values", "Classifications"], axis = 1, inplace = True)
    
FullTexts = DF.loc[DF.ChunkVol.eq("Chunk Vol: 0")].Text.unique()
Classifiers = DF.Classifier.unique()
    
Meta = []
Meta_Dicts = []      
for text in FullTexts:
    text = text.replace(".txt", "") #Remove ending, so that the text string can be identically found in both chunks and full texts
    TextSet = DF.loc[DF.Text.str.contains(text)] #Extract all columns from the overall DataFrame, which contain the string we've just created
    for classifier in Classifiers:
        ClassifierSet = TextSet.loc[TextSet.Classifier == classifier]
        x = list(ClassifierSet.Classification)
        y = dict(Counter(x))   
            
        summary = (text, classifier)
        Meta.append(summary)
        Meta_Dicts.append(y)
    
    
Meta_DF = pd.DataFrame(Meta)
Dict_DF = pd.DataFrame(Meta_Dicts)
ToExport = pd.concat([Meta_DF, Dict_DF], axis = 1)
ToExport.fillna(value = 0, inplace = True)

DF.to_csv("/Users/nate/Desktop/KNN Test Fix Raw.csv")
ToExport.to_csv("/Users/nate/Desktop/KNN Z Counts Summary.csv")

########## VALIDATION OF K VALUES
import pandas as pd

def Classification_Check(row):
    TextVal = row["Text"]
    AuthorVal = row["Classification"]
    if AuthorVal in TextVal:
        return 1
    else:
        return 0
    
DF["Success"] = DF.apply(lambda row: Classification_Check(row), axis = 1)

k_vals = DF["Paramter 1"].unique()
for k in k_vals:
    ToAnalyse = ToAnalyse.loc[ToAnalyse["Paramter 1"] == k]

    Success = float(ToAnalyse["Success"].sum())
    Total = float(len(ToAnalyse))
    Percentage_Success = (Success/Total) * 100.0