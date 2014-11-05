"""
Created on Tue Oct 21 19:20:10 2014
@Homework4
@Data Mining
@author: Josh Bacon
@description: Homework 4.
    Explores Decision Trees, K-Fold Cross-Validation,
    Classification Accuracies, and Confusion Matrices.
    K-Nearest Neighbros and Naive Bayes methods are included as
    comparison to the decision tree algorithms.

NOTE: Documentation is subpar, not enough time. Assignment was a "weekly script"

First csv Data set auto-data.txt is a table of car details data.
    Attribute Column Number and their meanings: 
        mpg = 0   <- Classification Label
        cylinders = 1  
        displacement = 2 
        horsepower = 3
        weight = 4
        acceleration = 5
        modelYear = 6
        origin = 7
        carName = 8
        msrp = 9
Second csv dataset is titanic-data.txt, a table of titanic passenger survival details
    Class = 0 
    Adult/Child = 1
    Male/Female = 2
    Lived/Died = 3  <- Class label
Accuracy is determined by Stratified-K-Fold-Cross-Validation (And is probabably incorrect right now
"""

import numpy
import math
import random
import operator
import copy
import csv

def readCsvFileToTable(fileName):
    """
    Reads CSV File from file name
    Returns a table (List of Lists)
    Note: All values from CSV are strings.
    Checks if value is an int, float, or string,
    then appends the correct type
    """
    fileOpen = open(fileName)
    csvFile = csv.reader(fileOpen)
    table = []
    instanceCount = 0
    for row in csvFile:
        rowList = []
        for attribute in row:
            try:
                newAttribute = int(attribute)
            except ValueError:
                try:
                    newAttribute = float(attribute)
                except ValueError:
                    newAttribute = attribute
            rowList.append(newAttribute)
        instanceCount = instanceCount + 1
        table.append(rowList)
    return table

def getFrequencies(vals):
    """
    Param1: A list of numeric values
    Returns:
    A list of values which are all the unique values from the input list
    and a list of counts which is mapped by index values
    Vals is a raw list of numbers (such as an entire attribute column, ie msrp)
    The return values are sorted from low to high
    """
    vals.sort()
    values, counts = [], []
    for item in vals:
        if(item not in values):
            values.append(item)
            counts.append(1)
        else:
            counts[-1] += 1
    return values, counts

""" Helper Functions """
def getAttrColumn(table, attrIndex):
    """"
    Param1: A table of the data such as that generated from auto-data.txt
    Param2: An integer that is an attribute in the table param
    and attrIndex (an integer specifying the column to choose)
    Returns the list from attribute column specified 
    """
    columnData = []
    for row in table:
        columnData.append(row[attrIndex])
    return columnData

def findEuclideanDistance(row,instanceToClass):
    total = 0
    for i in range(len(row)):
        total = total + ((row[i] - instanceToClass[i])*(row[i] - instanceToClass[i]))
    return math.sqrt(total)

def findKNearestNeighbors(testingSet, focusAttrs, instanceToClass, k):
    instanceToClassNarrowedAttrs = []
    for item in focusAttrs:
        instanceToClassNarrowedAttrs.append(instanceToClass[item])
    rowDistances =[]
    for row in testingSet:
        rowInstanceNarrowedAttrs = []
        for item in focusAttrs:
            rowInstanceNarrowedAttrs.append(row[item])
        d = findEuclideanDistance(rowInstanceNarrowedAttrs, instanceToClassNarrowedAttrs)
        rowDistances.append((d, row))
    rowDistances.sort(key=operator.itemgetter(0))
    orderedNeighbors = []
    for row in rowDistances:
        orderedNeighbors.append(row[1])
    return orderedNeighbors[0:k]

def chooseNearestNeighbor(nearestNeighbors, labelAttr):
    values, counts = getFrequencies(getAttrColumn(nearestNeighbors, labelAttr))
    majorityLabel = values[counts.index(max(counts))]
    for neighbor in nearestNeighbors:
        if(neighbor[labelAttr] == majorityLabel):
            return neighbor

def classifyInstancesByKNN(trainingSet, testingSet, k, focusAttrList, labelAttr):
    """ Takes a training and testing table of data (Can be the same). 
    NOTE: --Discrete d
    ata must be converted into numeric representation
            (Otherwise don't include the attribute in the focusAttributes list)
          --focusAttrList is a list of the attributes from the tables to use to find nearest neighbors. 
    """
    focusAttrListTuples = []
    for attr in focusAttrList:
        listFromAttr = getAttrColumn(trainingSet,attr)
        minList = min(listFromAttr)
        maxList = max(listFromAttr)
        maxMinList = (maxList - minList) * 1.0
        focusAttrListTuples.append((minList, maxMinList, attr))
        
    testingSetNarrowedNormalized = []
    for row in testingSet:
        rowNarrowedNormalized = []
        for i in range(len(row)):
            if i in focusAttrList:
                rowNarrowedNormalized.append((row[i] - focusAttrListTuples[focusAttrList.index(i)][0]) / focusAttrListTuples[focusAttrList.index(i)][1])
            else:
                rowNarrowedNormalized.append(row[i])
        testingSetNarrowedNormalized.append(rowNarrowedNormalized);
        
    classifications = []
    for i in range(len(testingSetNarrowedNormalized)):
        nearestNeighbors = findKNearestNeighbors(testingSetNarrowedNormalized, focusAttrList, testingSetNarrowedNormalized[i], k)      
        nearestNeighbor = chooseNearestNeighbor(nearestNeighbors, labelAttr)
        nearestNeighborIndex = testingSetNarrowedNormalized.index(nearestNeighbor)
        neighborClassification = testingSet[nearestNeighborIndex][labelAttr]
        classifications.append(neighborClassification)
    return classifications


def classifyInstancesByNaiveBayes(trainingSet, testingSet, labelsList, focusAttrList, labelAttr):
    lenLabels = len(labelsList)
    lenTraining = len(trainingSet)
    lenTesting = len(testingSet) 
    
    #Creates list of probabilities for each Label (using training set)    
    probLabelsList = [ 0 for i in range(lenLabels)]
    for instance in trainingSet:
        if instance[labelAttr] in labelsList:
            probLabelsList[labelsList.index(instance[labelAttr])] += 1
    for i in range(lenLabels):
        probLabelsList[i] = probLabelsList[i] / float(lenTraining)
        
    classifications = []
    for instance in testingSet:
        listProbsLabelGivenXNumerator = []
        for labelIndex in range(len(labelsList)):
            probXGivenLabel = 1
            for i in range(len(instance)):
                if i in focusAttrList: 
                    countOfSameValGivenLabel = 0
                    numInstancesWithLabel = 0
                    for j in range(lenTraining):
                        if(trainingSet[j][labelAttr] == labelsList[labelIndex]):
                            numInstancesWithLabel += 1
                            if(instance[i] == trainingSet[j][i]):
                                countOfSameValGivenLabel += 1
                    if(numInstancesWithLabel != 0):
                        probValGivenLabel = countOfSameValGivenLabel / float(numInstancesWithLabel)
                    else:
                        probValGivenLabel = 0
                    probXGivenLabel = probXGivenLabel * float(probValGivenLabel)
            numerator = probXGivenLabel * float(probLabelsList[labelIndex])
            listProbsLabelGivenXNumerator.append(numerator)
        labelWithHighestProb = labelsList[listProbsLabelGivenXNumerator.index(max(listProbsLabelGivenXNumerator))]
        classifications.append(labelWithHighestProb)
    return classifications
    
def findEntropy(dataTable, labelAttr):
    labelsList, labelsCounts = getFrequencies(getAttrColumn(dataTable, labelAttr))
    labelsProbs = [0 for x in labelsList]
    lenSet = len(dataTable)
    entropy = 0
    for j in range(len(labelsList)):
        if(labelsCounts[j] != 0):
            probOfLabel = labelsCounts[j] / float(lenSet)
            entropy = entropy - (probOfLabel * float(math.log(probOfLabel, 2.0)))
    return entropy

def getAttrGain(trainingSet, focusAttr, labelAttr):
    attrCategories, countCategories = getFrequencies(getAttrColumn(trainingSet, focusAttr))
    countCategories = len(attrCategories)
    partitions = [[] for x in attrCategories]
    for instance in trainingSet:
        partitions[attrCategories.index(instance[focusAttr])].append(instance)
    eStart = findEntropy(trainingSet, labelAttr)
    eNew = 0
    for i in range(len(partitions)):
        countPart = len(partitions[i])
        eNewPart = (countPart / float(len(trainingSet))) * findEntropy(partitions[i], labelAttr)
        eNew += eNewPart
    gain = eStart - eNew
    return gain

def findLargestGain(trainingSet, focusAttrList, labelAttr):
    largestGainAttr = 0
    largestGain = 0
    for attr in focusAttrList:
        attrGain = getAttrGain(trainingSet, attr, labelAttr)
        if(attrGain > largestGain):
            largestGainAttr = attr
            largestGain = attrGain
    return largestGainAttr, largestGain

def getFrequenciesOfValues(vals, possibleValues):
    counts = [0 for x in possibleValues]
    for item in vals:
        counts[possibleValues.index(item)] += 1
    return counts

def buildDecisionTree(trainingSet, focusAttrsValuesList, labelsList, focusAttrList, labelAttr):
    if(focusAttrList == []):
        return []
    largestGainAttr, largestGain = findLargestGain(trainingSet, focusAttrList, labelAttr)
    if(largestGain == 0):
        return []
    focusAttrsRemaining = focusAttrList[:]
    focusAttrsRemaining.pop(focusAttrList.index(largestGainAttr))
    focusAttrValuesListRemaining = focusAttrsValuesList[:]
    focusAttrValuesListRemaining.pop(focusAttrList.index(largestGainAttr))
    decisionTree = []
    decisionTree.append(largestGainAttr)
    possibleValues = focusAttrsValuesList[focusAttrList.index(largestGainAttr)]
    countsValues = getFrequenciesOfValues(getAttrColumn(trainingSet, largestGainAttr), possibleValues)  
    valuesDetailsList = []
    
    for i in range(len(possibleValues)):
        if(countsValues[i] != 0):
            instancesWithValue = []
            for instance in trainingSet:
                if(instance[largestGainAttr] == possibleValues[i]):
                   instancesWithValue.append(instance)    
            subTree = buildDecisionTree(instancesWithValue, focusAttrValuesListRemaining, labelsList, focusAttrsRemaining, labelAttr)
            if(subTree == []):
                #EndNode
                probabilitiesOfLabels = [[] for x in labelsList]
                classLabelCounts = getFrequenciesOfValues(getAttrColumn(instancesWithValue, labelAttr),labelsList)
                for j in range(len(probabilitiesOfLabels)):
                    probabilitiesOfLabels[j] = [labelsList[j], classLabelCounts[j] / float(len(instancesWithValue))]
                valuesDetailsList.append([possibleValues[i], probabilitiesOfLabels, []])
            else:
                valuesDetailsList.append([possibleValues[i], [], subTree])
    decisionTree.append(valuesDetailsList)
    return decisionTree

def printTreeRules(decisionTree, previousStr):
    if(decisionTree != []):
        
        for valueDetails in decisionTree[1]:
            newString = previousStr
            if(newString == ""):
                newString = "IF "
            newString += "attrIndex"+str(decisionTree[0]) + " == "+str(valueDetails[0])      
            if(valueDetails[1] != []):
                classLabel = 0
                classLabelProb = 0
                for classLabelDetails in valueDetails[1]:
                    if(classLabelDetails[1] > classLabelProb):
                        classLabel = classLabelDetails[0]
                        classLabelProb = classLabelDetails[1]
                newString += " THEN classLabel == "+str(classLabel) +" AND prob == "+str(classLabelProb)
                print newString
            else:
                newString += " AND "
                printTreeRules(valueDetails[2], newString)

def classifyInstanceByDecisionTree(instance, decisionTree, focusAttrList, labelsList, labelAttr):
    tree = copy.deepcopy(decisionTree)
    for i in range(len(focusAttrList)):
        attrOfTree = tree[0]
        attrValuesDetailsFromTree = tree[1]
        attrValueDetailsSelected = []
        for k in range(len(attrValuesDetailsFromTree)):
            attrValueFromTree = attrValuesDetailsFromTree[k]
            if(instance[attrOfTree] == attrValueFromTree[0]):
                attrValueDetailsSelected = attrValueFromTree
                break
        if(attrValueDetailsSelected != []):
            if(attrValueDetailsSelected[1] != []):
                greatestProbability = 0
                classLabelSelected = 0
                for classLabelProbability in attrValueDetailsSelected[1]:
                    if(classLabelProbability[1] > greatestProbability):
                        greatestProbability = classLabelProbability[1]
                        classLabelSelected = classLabelProbability[0]
                return classLabelSelected
        else:
            #Find class label with highest % between all the values under this attribute,
            # because the instance value is not in the tree

            classLabelAvgProbabilities= [[x,0] for x in labelsList]
            numClassLabels = len(classLabelAvgProbabilities)
            for attrValueDetails in attrValuesDetailsFromTree:
                #print str(classLabelAvgProbabilities) + "----" +str(attrValueDetails[1])
                for p in range(len(attrValueDetails[1])):
                    classLabelAvgProbabilities[p][1] += attrValueDetails[1][p][1]
            classLabelIndexHighestAvgProb = 0
            highestAvgProb = 0
            for j in range(len(classLabelAvgProbabilities)):
                classLabelAvgProbabilities[j][1] = classLabelAvgProbabilities[j][1] / float(numClassLabels)
                if(classLabelAvgProbabilities[j][1] > highestAvgProb):
                    classLabelIndexHighestAvgProb = classLabelAvgProbabilities[j][0]
                    highestAvgProb = classLabelAvgProbabilities[j][1]
            return classLabelIndexHighestAvgProb
                
        tree = attrValueDetailsSelected[2]
            

def classifyInstancesByDecisionTree(trainingSet, testingSet, focusAttrsValuesList, labelsList, focusAttrList, labelAttr, shouldPrintTreeRules):
    largestGainAttr, largestGain = findLargestGain(trainingSet, focusAttrList, labelAttr)
    decisionTree = buildDecisionTree(trainingSet, focusAttrsValuesList, labelsList, focusAttrList, labelAttr) 
    if(shouldPrintTreeRules):
        print "====================================================="
        print " Printing Decision Tree Rules (For Matrix Below)"
        print "====================================================="
        printTreeRules(decisionTree, "")
    classifications = []
    for instance in testingSet:
        classification = classifyInstanceByDecisionTree(instance, decisionTree, focusAttrList, labelsList, labelAttr)
        classifications.append(classification)
    return classifications

def convertTitanicDataToNumeric(titanicTable):
    """ Converts the catagorical data to numeric representation
        Required to calculate the distance from my KNN algorithm
    """
    numericTable = []
    for i in range(len(titanicTable)):
        numericRow = []
        if(titanicTable[i][0] == "crew"):
            numericRow.append(0)
        elif(titanicTable[i][0] == "third"):
            numericRow.append(1)
        elif(titanicTable[i][0] == "second"):
            numericRow.append(2)
        elif(titanicTable[i][0] == "first"):
            numericRow.append(3)
        if(titanicTable[i][1] == "adult"):
            numericRow.append(0)
        elif(titanicTable[i][1] == "child"):
            numericRow.append(1)
        if(titanicTable[i][2] == "male"):
            numericRow.append(0)
        elif(titanicTable[i][2] == "female"):
            numericRow.append(1)
        if(titanicTable[i][3] == "no"):
            numericRow.append(0)
        elif(titanicTable[i][3] == "yes"):
            numericRow.append(1)
        numericTable.append(numericRow)
    return numericTable
    
def covertTitanicDataToString(titanicTable):
    stringTable = []
    for i in range(len(titanicTable)):
        stringRow = []
        if(titanicTable[i][0] == 0):
            stringRow.append("crew")
        elif(titanicTable[i][0] == 1):
            stringRow.append("third")
        elif(titanicTable[i][0] == 2):
            stringRow.append("second")
        elif(titanicTable[i][0] == 3):
            stringRow.append("first")
        if(titanicTable[i][1] == 0):
            stringRow.append("adult")
        elif(titanicTable[i][1] == 1):
            stringRow.append("child")
        if(titanicTable[i][2] == 0):
            stringRow.append("male")
        elif(titanicTable[i][2] == 1):
            stringRow.append("female")
        if(titanicTable[i][3] == 0):
            stringRow.append("no")
        elif(titanicTable[i][3] == 1):
            stringRow.append("yes")
        stringTable.append(stringRow)
    
def getRatingForMPG(val):
        if val <= 13:
            return 1
        elif val == 14:
            return 2
        elif val <= 16:
            return 3
        elif val <= 19:
            return 4
        elif val <= 23:
            return 5
        elif val <= 26:
            return 6
        elif val <= 30:
            return 7
        elif val <= 36:
            return 8
        elif val <= 44:
            return 9
        else:
            return 10
            
def convertMpgToRating(mpgTable):
    newTable = []
    for instance in mpgTable:
        instance[0] = getRatingForMPG(instance[0])
        newTable.append(instance)
    return newTable

def getRankingForWeight(contWeightVal):
    if contWeightVal <= 1999:
        return 1
    elif contWeightVal < 2500:
        return 2
    elif contWeightVal < 3000:
        return 3
    elif contWeightVal < 3500:
        return 4
    elif contWeightVal >= 3500:
        return 5
    else:
        return 0

def discretizeWeightOfAutoData(mpgTable):
    newTable = []
    for instance in mpgTable:
        instance[4] = getRankingForWeight(instance[4])
        newTable.append(instance)
    return newTable
    
def calcAccuracyStratKFold(dataTable, kFolds, labelsList, labelAttr, classificationMethodDetails):
    """
        dataTable must be numeric for the focused attributes selected list
        Returns accuracy and Std Error
    """
    
    #Randomize dataTable
    randomizedTable = dataTable[:]
    numInstances = len(dataTable)
    for i in range(numInstances):
        j = random.randint(0, numInstances - 1)
        randomizedTable[i], randomizedTable[j] = randomizedTable[j], randomizedTable[i]
    #Partition Data by Classification attribute
    dataPartitions = [[] for x in range(len(labelsList))]
    for instance in randomizedTable:
        dataPartitions[labelsList.index(instance[labelAttr])].append(instance)
    #Partition data into Folds
    folds = [[] for x in range(kFolds)]
    for partition in dataPartitions:
        currentFold = 0
        for i in range(len(partition)):
            folds[currentFold].append(partition[i])
            currentFold += 1
            if(currentFold == kFolds):
                currentFold = 0
    #LabelMatrixCounts = [[0 for x in labelsList] for x in labelsList]
    accList = []
    stdErrorList = []
    for i in range(kFolds):
        #Get testing and training
        testingSet = folds[i]
        trainingSet = []
        for j in range(len(folds)):
            if(j != i):
                trainingSet = trainingSet + folds[j]
                
        if(classificationMethodDetails[0] == "kNN"):                
            classifications = classifyInstancesByKNN(trainingSet, testingSet, classificationMethodDetails[1], classificationMethodDetails[2], labelAttr)
        elif(classificationMethodDetails[0] == "naiveBayes"):
            classifications = classifyInstancesByNaiveBayes(trainingSet, testingSet, labelsList, classificationMethodDetails[1], labelAttr)
        elif(classificationMethodDetails[0] == "decisionTree"):
            classifications = classifyInstancesByDecisionTree(trainingSet, testingSet, classificationMethodDetails[2], labelsList, classificationMethodDetails[1], labelAttr, False)
        
        
        numInstances = 0
        numSuccessfulPredictions = 0
        for i in range(len(classifications)):
            numInstances += 1
            #print "Predicted: "+str(classifications[i])+ "    Actual: "+str(testingSet[i][labelAttr])
            if(classifications[i] == testingSet[i][labelAttr]):
                numSuccessfulPredictions += 1
        accuracy = numSuccessfulPredictions / float(numInstances)
        stdError = math.sqrt(accuracy *(1 - float(accuracy)) / float(len(testingSet)))
        accList.append(accuracy)
        stdErrorList.append(stdError)
        
    totalAcc = 0
    totalStdError = 0
    for i in range(len(accList)):
        totalAcc = totalAcc + accList[i]
        totalStdError = totalStdError + stdErrorList[i]
    accuracy = totalAcc / float(len(accList))
    stdError = totalStdError / float(len(stdErrorList))
    return accuracy, stdError

def printConfusionMatrixAutoData(mpgNumeric, classifications, title):
    print "==============================================="
    print title
    print "==============================================="
    labelsList = [1,2,3,4,5,6,7,8,9,10]
    labelCountsMatrix = [[0 for x in labelsList] for x in labelsList]
    labelAttr = 0
    for i in range(len(mpgNumeric)):
        labelCountsMatrix[labelsList.index(mpgNumeric[i][labelAttr])][labelsList.index(classifications[i])] += 1
    print "=====  ===  ===  ===  ===  ===  ===  ===  ===  ===  ===  =======  ==============="
    print "  MPG    1    2    3    4    5    6    7    8    9   10    Total   % Recognition"
    print "=====  ===  ===  ===  ===  ===  ===  ===  ===  ===  ===  =======  ==============="
    for i in range(len(labelCountsMatrix)):
        total = 0
        totalRecognized = 0
        string = str(i+1).rjust(5)
        for j in range(len(labelCountsMatrix)):
            string += str(labelCountsMatrix[i][j]).rjust(5)
            total = total + labelCountsMatrix[i][j]
            if(i == j):
                totalRecognized = totalRecognized + labelCountsMatrix[i][j]
        string += str(total).rjust(9)
        if(total != 0):
            recognition = totalRecognized / float(total)
        else:
            recognition = 0
        string += str(recognition).rjust(18)
        print string
    print "=====  ===  ===  ===  ===  ===  ===  ===  ===  ===  ===  =======  ==============="



def printConfusionMatrixTitanicData(titanicNumeric, classifications, title):
    print "==============================================="
    print title
    print "==============================================="
    labelsList = [0,1]
    labelCountsMatrix = [[0 for x in labelsList] for x in labelsList]
    labelAttr = 3
    for i in range(len(titanicNumeric)):
        labelCountsMatrix[labelsList.index(titanicNumeric[i][labelAttr])][labelsList.index(classifications[i])] += 1

    print "=====  ===  ===  =======  ==============="
    print "Lived  yes   no    Total    % Recognition"
    print "=====  ===  ===  =======  ==============="
    for i in range(len(labelCountsMatrix)):
        total = 0
        totalRecognized = 0
        string = str(i+1).rjust(5)
        for j in range(len(labelCountsMatrix)):
            string += str(labelCountsMatrix[i][j]).rjust(5)
            total = total + labelCountsMatrix[i][j]
            if(i == j):
                totalRecognized = totalRecognized + labelCountsMatrix[i][j]
        string += str(total).rjust(9)
        if(total != 0):
            recognition = totalRecognized / float(total)
        else:
            recognition = 0
        string += str(recognition).rjust(18)
        print string
    print "=====  ===  ===  ===  ===  ===  ===  ===  ===  ===  ===  =======  ==============="
        
        
def main():
    mpg = 0
    cylinders = 1  
    displacement = 2 
    horsepower = 3
    weight = 4
    acceleration = 5
    modelYear = 6
    origin = 7
    carName = 8
    msrp = 9
    
    classes = 0
    age = 1
    gender = 2
    lived = 3

    mpgTable = readCsvFileToTable("auto-data.txt") 
    titanicTable = readCsvFileToTable("titanic-data.txt")    
    
    #convert data columns to numeric
    titanicNumeric = convertTitanicDataToNumeric(titanicTable)
    mpgNumeric = convertMpgToRating(mpgTable)

    print "==============================================="
    print " Predictive Accuracy: "
    print "==============================================="
    print "   Stratified 10-fold Cross Validation:"
    print "      Titanic Data:"
    titanicKNNAccuracy, titanicKNNStdError = calcAccuracyStratKFold(titanicNumeric, 10, [0,1], lived, ["kNN", 50, [classes, age, gender]])
    print "         Top-K Nearest Neighbor: p = " +str(titanicKNNAccuracy)+ " +- " +str(titanicKNNStdError)
    titanicNBAccuracy, titanicNBStdError = calcAccuracyStratKFold(titanicNumeric, 10, [0,1], lived, ["naiveBayes", [classes, age, gender]])
    print "         Naive Bayes: p = " +str(titanicNBAccuracy)+ " +- " +str(titanicNBStdError)
    titanicDTAccuracy, titanicDTStdError = calcAccuracyStratKFold(titanicNumeric, 10, [0,1], lived, ["decisionTree", [classes, age, gender], [[0,1,2,3],[0,1],[0,1]]])
    print "         Decision Tree: p = " +str(titanicDTAccuracy)+ " +- " +str(titanicDTStdError)
    print " "
    print "      Auto Data:"
    mpgKNNAccuracy, mpgKNNStdError = calcAccuracyStratKFold(mpgNumeric, 10, [1,2,3,4,5,6,7,8,9,10], mpg, ["kNN", 10, [cylinders, weight, acceleration]])
    print "         Top-K Nearest Neighbor: p = " +str(mpgKNNAccuracy)+ " +- " +str(mpgKNNStdError)
    mpgNBAccuracy, mpgNBStdError = calcAccuracyStratKFold(mpgNumeric, 10, [1,2,3,4,5,6,7,8,9,10], mpg, ["naiveBayes",[cylinders, weight, acceleration]])
    print "         Naive Bayes: p = " +str(mpgNBAccuracy)+ " +- " +str(mpgNBStdError)
    mpgNumericDiscretized = discretizeWeightOfAutoData(mpgNumeric)
    modelYearValues, modelYearCounts = getFrequencies(getAttrColumn(mpgNumeric, modelYear))
    cylinderValues, cylinderValCounts = getFrequencies(getAttrColumn(mpgNumeric, cylinders))
    mpgDTAccuracy, mpgDTStdError = calcAccuracyStratKFold(mpgNumericDiscretized, 10, [1,2,3,4,5,6,7,8,9,10], mpg, ["decisionTree",[cylinders, weight, modelYear], [cylinderValues,[1,2,3,4,5],modelYearValues]])
    print "         Decision Tree: p = " +str(mpgDTAccuracy)+ " +- " +str(mpgDTStdError)
    

    classifications = classifyInstancesByKNN(mpgNumeric, mpgNumeric, 10, [cylinders, weight, acceleration], mpg)
    printConfusionMatrixAutoData(mpgNumeric, classifications, "Confusion Matrix: kNN Auto Data")    
    
    modelYearValues, modelYearCounts = getFrequencies(getAttrColumn(mpgNumeric, modelYear))
    cylinderValues, cylinderValCounts = getFrequencies(getAttrColumn(mpgNumeric, cylinders))
    classifications = classifyInstancesByDecisionTree(mpgNumeric, mpgNumeric, [cylinderValues,[1,2,3,4,5],modelYearValues], [1,2,3,4,5,6,7,8,9,10], [cylinders, weight, modelYear], mpg, True)
    printConfusionMatrixAutoData(mpgNumeric, classifications, "Confusion Matrix: Decision Tree Auto Data (Discretized Weight)")
    
    classifications = classifyInstancesByNaiveBayes(mpgNumeric, mpgNumeric, [1,2,3,4,5,6,7,8,9,10], [cylinders, weight, modelYear], mpg)
    printConfusionMatrixAutoData(mpgNumeric, classifications, "Confusion Matrix: Naive Bayes Auto Data")    
    
    
    classifications = classifyInstancesByKNN(titanicNumeric, titanicNumeric, 10, [0,1,2], 3)
    printConfusionMatrixTitanicData(titanicNumeric, classifications, "Confusion Matrix: kNN Titanic Data")
    
    classifications = classifyInstancesByNaiveBayes(titanicNumeric, titanicNumeric, [0,1], [0,1,2], 3)
    printConfusionMatrixTitanicData(titanicNumeric, classifications, "Confusion Matrix: Naive Bayes Titanic Data")
    
    classifications = classifyInstancesByDecisionTree(titanicNumeric, titanicNumeric, [[0,1,2,3],[0,1],[0,1]], [0,1], [0,1,2], 3, True)
    printConfusionMatrixTitanicData(titanicNumeric, classifications, "Confusion Matrix: Decision Tree Titanic Data")
    
    
    
main()


















