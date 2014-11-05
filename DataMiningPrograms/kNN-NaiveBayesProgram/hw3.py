"""
@authors Josh Bacon, Kelly 
@date 9/18/14
@description:
NOTE: Documentation is subpar, not enough time. Assignment was a "script" given a week.

This program performs kNearestNeighbor, LinearRegression, and Naive-Bayes Classification on the text file datasets.
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
Accuracy is determined by Stratified-K-Fold-Cross-Validation (And is probabably incorrect right now)

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
    
def getRatingsForMPG(vals):
    """
    NOTE: USED SOLELY for MPG COLUMN DATA
    Param1: a list of numeric values for the MPG
    Returns a list of equal length that maps each item from vals to a rating 0-10
    """
    ratings = []
    for item in vals:
        if item <= 13:
            ratings.append(1)
        elif item == 14:
            ratings.append(2)
        elif item <= 16:
            ratings.append(3)
        elif item <= 19:
            ratings.append(4)
        elif item <= 23:
            ratings.append(5)
        elif item <= 26:
            ratings.append(6)
        elif item <= 30:
            ratings.append(7)
        elif item <= 36:
            ratings.append(8)
        elif item <= 44:
            ratings.append(9)
        else:
            ratings.append(10)
    return ratings

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

def getRankingForWeight(val):
    if val <= 1999:
        return 1
    elif val < 2500:
        return 2
    elif val < 3000:
        return 3
    elif val < 3500:
        return 4
    elif val >= 3500:
        return 5
    else:
        return 0
    

    
   
def calcMeanValue(vals):
    """
    Param1: List of values without NA values
    Returns a float for the average of the list
    """
    total = 0
    for item in vals:
        total = total + item
    mean = total / float(len(vals))
    return mean
   
def findSlope(xVals, yVals):
    """
    Param1: a list of numeric x values
    Param2: a list of numeric y values 
    Note: Must be same length
    Returns a float representing the slope of best fitted line
    """
    xAvg = calcMeanValue(xVals)
    yAvg = calcMeanValue(yVals)
    numerator = 0
    denominator = 0
    for i in range(len(xVals)):
        numerator = numerator + ((xVals[i] - xAvg) * (yVals[i] - yAvg))
        denominator = denominator + (xVals[i] - xAvg) * (xVals[i] - xAvg)
    slope = numerator / denominator
    return slope
    
def findIntercept(slope, xVal, yVal):
    """"
    Param1: A float for the slope of line
    Param2: A xval representing a point on the line
    Param3: A yval representing a point on the line
    Returns the numeric y intercept value
    """
    return yVal - slope * xVal
    
def findCorrelationCoeff(xVals, yVals, xAvg, yAvg):
    """
    Param1: A list of numeric x values
    Param2: A list of numeric y values (Same length of X)
    Param3: The average of the x list passed in
    Param4: The average of the y passed in
    Returns a float representing the Correlation Coefficient
    """
    numerator = 0.0
    totalXDistFromMeanSqrd = 0.0
    totalYDistFromMeanSqrd = 0.0
    denominator = 0.0
    r = 0.0
    for i in range(len(xVals)):
        xDistFromAvg = (xVals[i] - xAvg)
        yDistFromAvg = (yVals[i] - yAvg)
        distancesMultiplied = xDistFromAvg * float(yDistFromAvg)
        numerator = numerator + distancesMultiplied
        xDistFromAvgSqrd = xDistFromAvg * float(xDistFromAvg)
        yDistFromAvgSqrd = yDistFromAvg * float(yDistFromAvg)
        totalXDistFromMeanSqrd = totalXDistFromMeanSqrd + xDistFromAvgSqrd
        totalYDistFromMeanSqrd = totalYDistFromMeanSqrd + yDistFromAvgSqrd
    denominator = math.sqrt(totalXDistFromMeanSqrd * float(totalYDistFromMeanSqrd))
    r = numerator / float(denominator)
    return r


def printInstance(table, listPredictions, title, convertMPG):
    print "=================================="
    print title
    print "=================================="
    for i in range(len(table)):
        print "instance: "+str(table[i][0])+", "+str(table[i][1])+", "+str(table[i][2])+", "+str(table[i][3])+", "+str(table[i][4])+", "+str(table[i][5])+", "+str(table[i][6])+", "+str(table[i][7])+", "+str(table[i][8])+", "+str(table[i][9])
        if(convertMPG == True):
            print "class: "+str(listPredictions[i])+", actual: "+str(getRatingForMPG(table[i][0]))
        else:
            print "class: "+str(listPredictions[i])+", actual: "+str(table[i][0])
            
def createRandomInstanceTable(table):
    lenTable = len(table)
    numAttr = len(table[0])
    randomInstances = []
    for i in range(10):
        randIndex = random.randint(0, lenTable - 1)
        randomInstances.append(table[randIndex])
    return randomInstances
        
def performLinearRegClassification(trainingSet, testingSet, attr1, attr2):
    """
    Performs Linear Regression Classification on a table with two selected attributes from the table
    """
    attr1List = getAttrColumn(trainingSet, attr1)
    attr2List = getAttrColumn(trainingSet, attr2)
    
    slope = findSlope(attr2List, attr1List)
    attr2ListAvg = calcMeanValue(attr2List)
    attr1ListAvg = calcMeanValue(attr1List)
    yIntercept = findIntercept(slope, attr2ListAvg, attr1ListAvg)

    classifications = []
    for item in testingSet:
        mpgPrediction = item[attr2] * float(slope) + yIntercept
        mpgClassification = getRatingForMPG(mpgPrediction)
        classifications.append(mpgClassification)
    return testingSet, classifications
    
def kNNClassifier(testingSet, numAttr, instanceToClass, k):
    rowDistances =[]
    for row in testingSet:
        d = findEuclideanDistance(row, instanceToClass)
        rowDistances.append([d, row])
    
    rowDistances.sort(key=operator.itemgetter(0))
    label = rowDistances[1] #The first one is always going to be itself
    return label

def findEuclideanDistance(row,instanceToClass):
    total = 0
    for i in range(len(row)):
        total = total + ((row[i] - instanceToClass[i])*(row[i] - instanceToClass[i]))
    return math.sqrt(total)

def performKNearestNeighbor(trainingSet, testingSet, classifiedAttrIndex, k, listOfAttrIndex):
    #Training Set to determine how to normalize
    listOfAttrTuples = []
    for index in listOfAttrIndex:
        listFromAttr = getAttrColumn(trainingSet,index)
        minList = min(listFromAttr)
        maxList = max(listFromAttr)
        maxMinList = (maxList - minList) * 1.0
        listOfAttrTuples.append( (minList, maxMinList))
    
    #Normalize and focus the Testing Set
    normalizedFocusSet = []
    for row in testingSet:
        normalizedFocusRow = []
        for i in range(len(listOfAttrIndex)):
            listOfAttrTuples[i]
            listOfAttrIndex[i]
            normalizedFocusRow.append((row[listOfAttrIndex[i]] - listOfAttrTuples[i][0]) / listOfAttrTuples[i][1])
        normalizedFocusSet.append(normalizedFocusRow)
    nearestNeighborsNormalized = [] #same size and order
    indexesOfClassifiedNormalizedInstances = []
    for i in range(len(normalizedFocusSet)):
        indexesOfClassifiedNormalizedInstances.append(i)
        nearestNeighborNormalized = kNNClassifier(normalizedFocusSet,len(listOfAttrIndex), normalizedFocusSet[i], k)[1]
        nearestNeighborsNormalized.append(nearestNeighborNormalized)
    classifiedInstances = []
    classifications = []
    for i in range(len(indexesOfClassifiedNormalizedInstances)):
        classifiedIntanceFromTesting = testingSet[indexesOfClassifiedNormalizedInstances[i]]
        classifiedInstances.append(classifiedIntanceFromTesting)
        classificationOfInstance = getRatingForMPG(testingSet[normalizedFocusSet.index(nearestNeighborsNormalized[i])][classifiedAttrIndex])
        classifications.append(classificationOfInstance)        
        
    return classifiedInstances, classifications

def performNaiveBayes1(table):
    mpg = 0
    cylinders = 1  
    weight = 4
    modelYear = 6
    
    numInstances = len(table)

    trainingSet = copy.deepcopy(table)

    randomInstances = createRandomInstanceTable(trainingSet)

    for i in range(len(trainingSet)):
        trainingSet[i][weight] = getRankingForWeight(trainingSet[i][weight])
        trainingSet[i][mpg] = getRatingForMPG(trainingSet[i][mpg])

    cylinderValues, cylinderCounts = getFrequencies(getAttrColumn(trainingSet, cylinders))    
    weightValues, weightCounts = getFrequencies(getAttrColumn(trainingSet, weight))    
    modelYearValues, modelYearCounts = getFrequencies(getAttrColumn(trainingSet, modelYear))
    mpgRankValues,mpgRankCounts = getFrequencies(getAttrColumn(trainingSet, mpg))

    rowsMpgRanks = [[]for i in range(10)]
    for instance in trainingSet:
        if(instance[mpg] == 1):
            rowsMpgRanks[0].append(instance)
        elif(instance[mpg] == 2):
            rowsMpgRanks[1].append(instance)
        elif(instance[mpg] == 3):
            rowsMpgRanks[2].append(instance)
        elif(instance[mpg] == 4):
            rowsMpgRanks[3].append(instance)
        elif(instance[mpg] == 5):
            rowsMpgRanks[4].append(instance)
        elif(instance[mpg] == 6):
            rowsMpgRanks[5].append(instance)
        elif(instance[mpg] == 7):
            rowsMpgRanks[6].append(instance)
        elif(instance[mpg] == 8):
            rowsMpgRanks[7].append(instance)
        elif(instance[mpg] == 9):
            rowsMpgRanks[8].append(instance)
        elif(instance[mpg] == 10):
            rowsMpgRanks[9].append(instance)
            
    pHList = []
    for i in range(9):
        pHList.append(mpgRankCounts[i] / float(numInstances))


    classifications = []
    for instance in randomInstances:
        
        probCylinder = cylinderCounts[cylinderValues.index(instance[cylinders])] / float(numInstances)

        probWeight = weightCounts[weightValues.index(instance[weight])] / float(numInstances)
        probModelYear = modelYearCounts[modelYearValues.index(instance[modelYear])] / float(numInstances)

        pX = probCylinder * probWeight * float(probModelYear)

        probClassList = []
        
        for i in range(9):
            cylinderValuesH,cylinderCountsH = getFrequencies(getAttrColumn(rowsMpgRanks[i],cylinders))
            weightValuesH, weightCountsH = getFrequencies(getAttrColumn(rowsMpgRanks[i], weight))    
            modelYearValuesH, modelYearCountsH = getFrequencies(getAttrColumn(rowsMpgRanks[i], modelYear))
            if instance[cylinders] in cylinderValuesH:
                probCylinderH = cylinderCountsH[cylinderValuesH.index(instance[cylinders])] / float(numInstances)
            else:
                probCylinderH = 0
            if instance[weight] in weightValuesH:
                probWeightH = weightCountsH[weightValuesH.index(instance[weight])] / float(numInstances)
            else:
                probWeightH = 0
            if instance[modelYear] in modelYearValuesH:
                probModelYearH = modelYearCountsH[modelYearValuesH.index(instance[modelYear])] / float(numInstances)                                             
            else:
                probModelYearH = 0
            pXH = probCylinderH * probWeightH * float(probModelYearH)
            numerator = pXH * float(pHList[i])
            demoninator = pX
            probClassList.append([ numerator / float(demoninator)])
        classifications.append(mpgRankValues[probClassList.index(max(probClassList))])

    printInstance(randomInstances, classifications, "STEP 3(Part 1): Naive Bayes MPG Classifiers", False)

def gaussian(x, mean, sdev):
    first, second = 0,0
    if sdev > 0:
        first = 1 / (math.sqrt(2 * math.pi) * sdev)
        second = math.e ** (-((x - mean) ** 2) / ( 2 * (sdev ** 2)))
    return first * second

#2:1 Partition
def holdout_partition(table):
    randomized = table[:]
    n = len(table)
    for i in range(n):
        #pick an index to swap
        j = random.randint(0, n - 1)   #random int [0,  n-1] inclusive
        randomized[i],randomized[j] = randomized[j], randomized[i]
    #return train and test sets
    n0 = (n * 2) / 3
    return randomized[0:n0], randomized[n0:]

def RandomSubSamples(table, k, attrLabelIndex, formulaString):
    cylinders = 1
    weight = 4
    acceleration = 5
    accuracyList = []
    
    labelsList = [1,2,3,4,5,6,7,8,9,10]

    labelMatrixCounts = [[0 for x in labelsList] for x in labelsList]    
    
    for i in range(k):
        training, testing = holdout_partition(table)
        if(formulaString == "Linear Regression"):
            classifiedInstances, classifications = performLinearRegClassification(training, testing, attrLabelIndex, weight)      
        elif(formulaString == "Naive Bayes 1"):
            classifiedInstances, classifications = performLinearRegClassification(training, testing, attrLabelIndex, weight)
        elif(formulaString == "Naive Bayes II"):
            classifiedInstances, classifications = performLinearRegClassification(training, testing, attrLabelIndex, weight)
        elif(formulaString == "Top-K Nearest Neighbor"):
            classifiedInstances, classifications = performKNearestNeighbor(training, testing, attrLabelIndex, 5, [cylinders, weight, acceleration])

        for q in range(len(testing)):
            mpgRankActual = getRatingForMPG(testing[q][attrLabelIndex])
            mpgRankPredict = classifications[q]
            labelMatrixCounts[mpgRankActual - 1][mpgRankPredict - 1] += 1
        
        #Find counts of each labels in classifications   
        labelCounts = [0 for i in range(len(labelsList))]        
        for i in range(len(classifications)):
            labelCounts[labelsList.index(classifications[i])] += 1
        #Finding weights of labels for accuracy averaging
        labelWeights = []
        for i in range(len(labelCounts)):
            labelWeights.append(labelCounts[i] / float(len(testing)))
            
        accuracyListPerLabel = []
        for i in range(len(labelsList)):
            numTruePos = 0
            numTrueNeg = 0
            numFalseNeg = 0
            numFalsePos = 0
            for j in range(len(testing)):
                actualLabel = getRatingForMPG(testing[j][attrLabelIndex])
                predictedLabel = classifications[j]
                if(actualLabel == i+1 and predictedLabel == i+1):
                    numTruePos += 1
                elif(actualLabel == i+1 and predictedLabel != i+1):
                    numFalseNeg += 1
                elif(actualLabel != i+1 and predictedLabel == i+1):
                    numFalsePos += 1
                elif(actualLabel != i+1 and predictedLabel != i+1):
                    numTrueNeg += 1

            accuracyForLabel = (numTruePos + numTrueNeg) / float(len(testing))
            #print "Label: "+str(i+1)+ "   Values: "+str(accuracyForLabel) + " Weight: "+str(labelWeights[i])
            accuracyListPerLabel.append((accuracyForLabel, labelWeights[i]))
        averageForIteration = 0
        for item in accuracyListPerLabel:
            averageForIteration += float(item[0]) * item[1]
        accuracyList.append(averageForIteration)
    totalAcc = 0
    for item in accuracyList:
        totalAcc += float(item)
    averageAccuracy = totalAcc / float(k)
    standardErr = math.sqrt(averageAccuracy *(1 - float(averageAccuracy)) / float(len(testing)))
    return averageAccuracy, standardErr, labelMatrixCounts

def StratifiedkFoldCrossValidation(table, k, attrLabelIndex, formulaString):
    cylinders = 1
    weight = 4
    acceleration = 5
    #Randomize and divide Data into k subsets
    randomized = table[:]
    n = len(table)
    for i in range(n):
        #pick an index to swap
        j = random.randint(0, n - 1)   #random int [0,  n-1] inclusive
        randomized[i],randomized[j] = randomized[j], randomized[i]
    #Partition data by classification attribute
    labelsList = [1,2,3,4,5,6,7,8,9,10]
    
    dataPartitions = [[] for x in range(len(labelsList))]
    
    for instance in randomized:
        dataPartitions[labelsList.index(getRatingForMPG(instance[attrLabelIndex]))].append(instance)
    #Make Folds by dividing up between dataPartitions
    folds = [[] for x in range(k)]
    for partition in dataPartitions:
        whichFold = 0
        for i in range(len(partition)):
            folds[whichFold].append(partition[i])
            whichFold += 1
            if(whichFold == k):
                whichFold = 0
    
    labelMatrixCounts = [[0 for x in labelsList] for x in labelsList]
    
    accuracyList = []  
    for i in range(k):
        #Get testing and training
        testing = folds[i]
        training = []
        for j in range(k):
            if(j != i):
                training = training + folds[j]
        if(formulaString == "Linear Regression"):
            classifiedInstances, classifications = performLinearRegClassification(training, testing, attrLabelIndex, weight)      
        elif(formulaString == "Naive Bayes 1"):
            classifiedInstances, classifications = performLinearRegClassification(training, testing, attrLabelIndex, weight)
        elif(formulaString == "Naive Bayes II"):
            classifiedInstances, classifications = performLinearRegClassification(training, testing, attrLabelIndex, weight)
        elif(formulaString == "Top-K Nearest Neighbor"):
            classifiedInstances, classifications = performKNearestNeighbor(training, testing, attrLabelIndex, 5, [cylinders, weight, acceleration])
        #for i in range(len(classifications)):
         #   classifications[i] = random.randint(1, 10)
            
        for q in range(len(testing)):
            mpgRankActual = getRatingForMPG(testing[q][attrLabelIndex])
            mpgRankPredict = classifications[q]
            labelMatrixCounts[mpgRankActual - 1][mpgRankPredict - 1] += 1
        
        #Find counts of each labels in classifications   
        labelCounts = [0 for i in range(len(labelsList))]        
        for i in range(len(classifications)):
            labelCounts[labelsList.index(classifications[i])] += 1
        #Finding weights of labels for accuracy averaging
        labelWeights = []
        for i in range(len(labelCounts)):
            labelWeights.append(labelCounts[i] / float(len(testing)))
        
        
        accuracyListPerLabel = []
        for i in range(len(labelsList)):
            numTruePos = 0
            numTrueNeg = 0
            numFalseNeg = 0
            numFalsePos = 0
            for j in range(len(testing)):
                actualLabel = getRatingForMPG(testing[j][attrLabelIndex])
                predictedLabel = classifications[j]
                if(actualLabel == i+1 and predictedLabel == i+1):
                    numTruePos += 1
                elif(actualLabel == i+1 and predictedLabel != i+1):
                    numFalseNeg += 1
                elif(actualLabel != i+1 and predictedLabel == i+1):
                    numFalsePos += 1
                elif(actualLabel != i+1 and predictedLabel != i+1):
                    numTrueNeg += 1

            accuracyForLabel = (numTruePos + numTrueNeg) / float(len(testing))
            #print "Label: "+str(i+1)+ "   Values: "+str(accuracyForLabel) + " Weight: "+str(labelWeights[i])
            accuracyListPerLabel.append((accuracyForLabel, labelWeights[i]))
        averageForIteration = 0
        for item in accuracyListPerLabel:
            averageForIteration += float(item[0]) * item[1]
        accuracyList.append(averageForIteration)
    totalAcc = 0
    for item in accuracyList:
        totalAcc += float(item)
    averageAccuracy = totalAcc / float(k)
    standardErr = math.sqrt(averageAccuracy *(1 - float(averageAccuracy)) / float(len(testing)))
    return averageAccuracy, standardErr, labelMatrixCounts

def performValidation(table):
    print "=============================="
    print "STEP 4: Predictive Accuracy"
    print "=============================="
    print "  Random Subsample (k = 10, 2:1 Train/Test) "
    accuracyResult, standardErr, countsMatrix = RandomSubSamples(table, 10, 0, "Linear Regression")
    print "     Linear Regression:      p = "+str(accuracyResult)+" +- "+str(standardErr)
    accuracyResult, standardErr, countsMatrix = RandomSubSamples(table, 10, 0, "Linear Regression")
    print "     Naive Bayes I:      p = "+str(accuracyResult)+" +- "+str(standardErr)
    accuracyResult, standardErr, countsMatrix = RandomSubSamples(table, 10, 0, "Linear Regression")
    print "     Naive Bayes II:      p = "+str(accuracyResult)+" +- "+str(standardErr)
    accuracyResult, standardErr, countsMatrix = RandomSubSamples(table, 10, 0, "Top-K Nearest Neighbor")
    print "     Top-K Nearest Neighbor: p = "+str(accuracyResult)+" +- "+str(standardErr)
    
    print "  Stratified K-Fold Cross Validation (k = 10)"
    accuracyResult, standardErr, countsMatrix = StratifiedkFoldCrossValidation(table, 10, 0, "Linear Regression")
    print "     Linear Regression:      p = "+str(accuracyResult)+" +- "+str(standardErr)
    accuracyResult, standardErr, countsMatrix = StratifiedkFoldCrossValidation(table, 10, 0, "Linear Regression")
    print "     Naive Bayes I:      p = "+str(accuracyResult)+" +- "+str(standardErr)
    accuracyResult, standardErr, countsMatrix = StratifiedkFoldCrossValidation(table, 10, 0, "Linear Regression")
    print "     Naive Bayes II:      p = "+str(accuracyResult)+" +- "+str(standardErr)
    accuracyResult, standardErr, countsMatrix = StratifiedkFoldCrossValidation(table, 10, 0, "Top-K Nearest Neighbor")
    print "     Top-K Nearest Neighbor: p = "+str(accuracyResult)+" +- "+str(standardErr)

def createConfusionMatrices(table):
    print "Linear Regression (Stratified 10-Fold Cross Validation):"
    print "=====  ===  ===  ===  ===  ===  ===  ===  ===  ===  ====  ======= ================="
    print "  MPG    1    2    3    4    5    6    7    8    9    10    Total   Recognition (%)"
    print "=====  ===  ===  ===  ===  ===  ===  ===  ===  ===  ====  ======= ================="    
    accuracyResult, standardErr, countsMatrix = StratifiedkFoldCrossValidation(table, 10, 0, "Linear Regression")
    for i in range(len(countsMatrix)):
        total = 0
        totalRecognized = 0
        string = str(i+1).rjust(5)
        for j in range(len(countsMatrix[i])):
            string += str(countsMatrix[i][j]).rjust(5)
            total = total + countsMatrix[i][j]
            if(i == j):
                totalRecognized = countsMatrix[i][j]
        string += str(total).rjust(9)
        if(total != 0):
            recognition =  totalRecognized / float(total)
        else:
            recognition = 0
        string += str(recognition).rjust(18)
        print string
    print "===== === === === === === === === === === ==== ======= ================="

def convertTitanicClassNumsToStrings(numericData):
    stringData = []
    for instance in numericData:
        if(instance == 1):
            stringData.append("yes")
        elif(instance == 2):
            stringData.append("no")
    return stringData

def convertTitanicClassToString(item):
    if(item == 1):
        return "yes"
    else:
        return "no"
def convertTitanicStringClassToInt(item):
    if(item == "yes"):
        return 1
    else:
        return 2
def convertTitaniStringDataToNumeriec(titanicTable):
    numericData = []
    for instance in titanicTable:
        numericDataRow = []
        if(instance[0] == "first"):
            numericDataRow.append(1)
        elif(instance[0] == "second"):
            numericDataRow.append(2)
        elif(instance[0] == "third"):
            numericDataRow.append(3)
        elif(instance[0] == "crew"):
            numericDataRow.append(4)
        if(instance[1] == "child"):
            numericDataRow.append(1)
        elif(instance[1] == "adult"):
            numericDataRow.append(2)
        if(instance[2] == "female"):
            numericDataRow.append(1)
        elif(instance[2] == "male"):
            numericDataRow.append(2)
        if(instance[3] == "yes"):
            numericDataRow.append(1)
        elif(instance[3] == "no"):
            numericDataRow.append(2)
        numericData.append(numericDataRow)
    return numericData
def convertTitanicNumerieDataToStrings(numericData):
    stringData = []
    for instance in numericData:
        stringDataRow = []
        if(instance[0] == 1):
            stringDataRow.append("first")
        elif(instance[0] == 2):
            stringDataRow.append("second")
        elif(instance[0] == 3):
            stringDataRow.append("third")
        elif(instance[0] == 4):
            stringDataRow.append("crew")
        if(instance[1] == 1):
            stringDataRow.append("child")
        elif(instance[1] == 2):
            stringDataRow.append("adult")
        if(instance[2] == 1):
            stringDataRow.append("female")
        elif(instance[2] == 2):
            stringDataRow.append("male")
        if(instance[3] == 1):
            stringDataRow.append("yes")
        elif(instance[3] == 2):
            stringDataRow.append("no")
        stringData.append(stringDataRow)
    return stringData

def doStratkFoldForTitanc(table) :
    k = 10
    attrLabelIndex = 3    
    
    randomized = table[:]
    n = len(table)
    for i in range(n):
        #pick an index to swap
        j = random.randint(0, n - 1)   #random int [0,  n-1] inclusive
        randomized[i],randomized[j] = randomized[j], randomized[i]
    #Partition data by classification attribute
    labelsList = [1, 2]
    
    dataPartitions = [[] for x in range(len(labelsList))]
    
    for instance in randomized:
        dataPartitions[labelsList.index(instance[attrLabelIndex])].append(instance)
    #Make Folds by dividing up between dataPartitions
    folds = [[] for x in range(k)]
    for partition in dataPartitions:
        whichFold = 0
        for i in range(len(partition)):
            folds[whichFold].append(partition[i])
            whichFold += 1
            if(whichFold == k):
                whichFold = 0
    
    labelMatrixCounts = [[0 for x in labelsList] for x in labelsList]
    
    accList = []
    for i in range(k):
        #Get testing and training
        testing = folds[i]
        training = []
        for j in range(k):
            if(j != i):
                training = training + folds[j]
        classifiedInstances, classifications = doKNNForTitanic(training, testing)

        for q in range(len(testing)):
            mpgRankActual = testing[q][attrLabelIndex]
            mpgRankPredict = convertTitanicStringClassToInt(classifications[q])
            labelMatrixCounts[mpgRankActual - 1][mpgRankPredict - 1] += 1
        
        
        
        #Accuracy calculated differently than the other ways
        numPredictedLived = 0
        numPredictedDied = 0
        totalPredicted = 0
        for instance in classifications:
            totalPredicted += 1
            if(instance == "yes"):
                numPredictedLived += 1
            else:
                numPredictedDied += 1
        numActualLived = 0
        numActualDied = 0
        for instance in getAttrColumn(testing, 3):
            if(convertTitanicClassToString(instance) == "yes"):
                numActualLived += 1
            else:
                numActualDied += 1
        predRatio = numPredictedLived / float(numPredictedLived + numPredictedDied)
        actRatio= numActualLived / float(numActualLived + numActualDied)
        percentOff = abs(predRatio - actRatio) / float(actRatio)
        accuracy = 1.0 - percentOff
        accList.append(accuracy)
    totalAcc = 0.0
    for a in accList:
        totalAcc += a
    averageAccuracy = totalAcc / float(k)
        
    standardErr = math.sqrt(averageAccuracy *(1 - float(averageAccuracy)) / float(len(testing)))
    return averageAccuracy, standardErr, labelMatrixCounts

def doKNNForTitanic(trainingSet, testingSet):
    classifiedAttrIndex = 3
    k = 10
    listOfAttrIndex = [0, 1, 2]
    #Training Set to determine how to normalize
    listOfAttrTuples = []
    for index in listOfAttrIndex:
        listFromAttr = getAttrColumn(trainingSet,index)
        minList = min(listFromAttr)
        maxList = max(listFromAttr)
        maxMinList = (maxList - minList) * 1.0
        listOfAttrTuples.append( (minList, maxMinList))
    
    #Normalize and focus the Testing Set
    normalizedFocusSet = []
    for row in testingSet:
        normalizedFocusRow = []
        for i in range(len(listOfAttrIndex)):
            listOfAttrTuples[i]
            listOfAttrIndex[i]
            normalizedFocusRow.append((row[listOfAttrIndex[i]] - listOfAttrTuples[i][0]) / listOfAttrTuples[i][1])
        normalizedFocusSet.append(normalizedFocusRow)
        
        
    nearestNeighborsNormalized = [] #same size and order
    indexesOfClassifiedNormalizedInstances = []
    for i in range(len(normalizedFocusSet)):
        indexesOfClassifiedNormalizedInstances.append(i)
        nearestNeighborNormalized = kNNClassifier(normalizedFocusSet,len(listOfAttrIndex), normalizedFocusSet[i], k)[1]
        nearestNeighborsNormalized.append(nearestNeighborNormalized)
    classifiedInstances = []
    classifications = []
    for i in range(len(indexesOfClassifiedNormalizedInstances)):
        classifiedIntanceFromTesting = testingSet[indexesOfClassifiedNormalizedInstances[i]]
        classifiedInstances.append(classifiedIntanceFromTesting)
        classificationOfInstance = testingSet[normalizedFocusSet.index(nearestNeighborsNormalized[i])][classifiedAttrIndex]
        classifications.append(classificationOfInstance)  
    classificationsLabeled = convertTitanicClassNumsToStrings(classifications)
    return classifiedInstances, classificationsLabeled

def doTitanicExample():
    titanicTable = readCsvFileToTable("titanic-data.txt")
    #Convert data to numeric
    numericData = []
    for instance in titanicTable:
        numericDataRow = []
        if(instance[0] == "first"):
            numericDataRow.append(1)
        elif(instance[0] == "second"):
            numericDataRow.append(2)
        elif(instance[0] == "third"):
            numericDataRow.append(3)
        elif(instance[0] == "crew"):
            numericDataRow.append(4)
        if(instance[1] == "child"):
            numericDataRow.append(1)
        elif(instance[1] == "adult"):
            numericDataRow.append(2)
        if(instance[2] == "female"):
            numericDataRow.append(1)
        elif(instance[2] == "male"):
            numericDataRow.append(2)
        if(instance[3] == "yes"):
            numericDataRow.append(1)
        elif(instance[3] == "no"):
            numericDataRow.append(2)
        numericData.append(numericDataRow)
    classifiedInstances, classificationsLabeled = doKNNForTitanic(numericData, createRandomInstanceTable(numericData))
    
    stringClassifiedInstances = convertTitanicNumerieDataToStrings(classifiedInstances)
    print"================================================="
    print" Titanic Data - kNN results for 5 random instances"
    print"================================================="
    for i in range(5):
        print "instance: "+str(stringClassifiedInstances[i][0])+", "+str(stringClassifiedInstances[i][1])+", "+str(stringClassifiedInstances[i][2])+", "+str(stringClassifiedInstances[i][3])
        print "class: "+str(classificationsLabeled[i])+", actual: "+str(stringClassifiedInstances[i][3])
 
    averageAccuracy, standardErr, labelMatrixCounts = doStratkFoldForTitanc(numericData)
    print "============================="
    print "Predictive Accuracy - Stratified 10-Fold Cross Validation"
    print "================================"
    print "Top-K Nearest Neighbor(Mod): p = "+str(averageAccuracy)+"  +-  "+str(standardErr)
    print "===== ====== ====== ====== =================="
    print " L/D  Lived   Die     Total   Recognition (%)"
    print "===== ====== ====== ====== ================="
    print "Live"+str(labelMatrixCounts[0][0]).rjust(8) + str(labelMatrixCounts[0][1]).rjust(8)+str(labelMatrixCounts[0][1]+labelMatrixCounts[0][0]).rjust(8)+str(labelMatrixCounts[0][0] / float(labelMatrixCounts[0][1] + labelMatrixCounts[0][0])).rjust(15)
    print " Die"+str(labelMatrixCounts[1][0]).rjust(8) + str(labelMatrixCounts[1][1]).rjust(8)+str(labelMatrixCounts[1][1]+labelMatrixCounts[1][0]).rjust(8).rjust(8)+str(labelMatrixCounts[1][1] / float(labelMatrixCounts[1][1] + labelMatrixCounts[1][0])).rjust(15)
 
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

    table = readCsvFileToTable("auto-data.txt") 
    #Step 1:
    classifiedInstances, classifications = performLinearRegClassification(table, createRandomInstanceTable(table), mpg, weight)
    printInstance(classifiedInstances, classifications,"STEP1: Linear Regression MPG Classifier", True)    
    #Step 2:
    classifiedInstances, classifications = performKNearestNeighbor(table, createRandomInstanceTable(table), mpg, 5, [cylinders, weight, acceleration])
    printInstance(classifiedInstances, classifications,"STEP 2: k=5 Nearest Neighbor MPG Classifier", True)    
    #Step 3:
    performNaiveBayes1(table)
    #performNaiveBayes2(table)
    #Step 4:
    performValidation(table)
    #Step 5:
    createConfusionMatrices(table)
    #Step 6:
    doTitanicExample()    
    
main()

##if __name__ == '__main__':
##    main()
