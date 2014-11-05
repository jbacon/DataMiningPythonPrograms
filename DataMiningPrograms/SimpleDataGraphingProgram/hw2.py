"""
@authors Josh Bacon, Kelly
@date 9/18/14
@description: Homework Project/Assignment 2: Experiments with basic data 
visualization techniques using the matplotlib Python module.
"""

import numpy
import matplotlib.pyplot as pyplot
import math
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


def groupBy(table, attrIndex):
    groupingValues = []
    for row in table:
        if row[attrIndex] not in groupingValues:
            groupingValues.append(row[attrIndex])
    groupingValues.sort()
    result = [ [] for _ in groupingValues]
    for row in table:
        result[groupingValues.index(row[attrIndex])].append(row[:])
    return result

def getEqualWidthBins(vals, numBins):
    """
    Param1: A list of numeric values
    Param2: A integer specifying the number of bins to divide values into
    Returns a list of cutoffs values at equal widths 
    indicating where bins should be divided
    Vals is just a raw list of numbers (such as an entire attribute column)
    """
    width = int(max(vals) - min(vals)) / numBins
    cutOffs = []
    cutOffs.append(int(min(vals) + width))
    for i in range(numBins - 2):
        cutOffs.append(cutOffs[-1] + width)
    return cutOffs

def discretize(vals, cutOffs):
    """
    Param1: A list of numeric values. 
    Param2: A list of numeric cutOff values of the list
    Returns a list of cutOffs+1 number of integers 
    which represent the number of values from vals that fall in the bins 
    indicated by the cutOff values Vals is just a raw list of numbers 
    (such as an entire attribute column)
    """
    newVals = []
    for v in vals:
        found = False
        for i in range(len(cutOffs)):
            if not found & v < cutOffs[i]:
                newVals.append(i + 1)
                found = True
        if not found:
            newVals.append(len(cutOffs + 1))
    return newVals

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
    
def makeHistogram(vals, xLabelVals, title, labelX, labelY, fileName):
    """
    Param1: A List of numeric values (which will be sorted into intervals)
    Param2: A list of strings which are the labels for different intervals
    Param3: A string for the title of the graph
    Param4: A string for the x axis label
    Param5: A string for the y axis label
    Saves a pdf of a histogram
    Description: Uses getFrequencies to automatically convert data and make xticks
    Vals is just a raw list of numbers (such as an entire attribute column)
    """
    vals, freqs = getFrequencies(vals)
    pyplot.figure()
    pyplot.title(title)
    pyplot.xlabel(labelX)
    pyplot.ylabel(labelY)
    xrng = numpy.arange(len(vals))
    if(xLabelVals == []):
        pyplot.xticks(xrng)
    else:
        pyplot.xticks(xrng, xLabelVals, rotation=30, size='small')
    yrng = numpy.arange(0, max(freqs), 10)
    pyplot.yticks(yrng)
    pyplot.bar(xrng, freqs, .45,  align = 'center', edgecolor = 'none')
    pyplot.grid(True)
    
    pyplot.savefig(fileName)
    pyplot.close()

    
def makePieChart(vals, title, fileName):
    """
    Param1: A list of numeric values
    Param2: A string
    Saves a pdf of a Pie Chart of the data
    Description: Uses getFrequencies to automatically make convert dataVals is just a
    raw list of numbers (such as an entire attribute column)
    """
    vals, freqs = getFrequencies(vals)
    pyplot.figure()
    pyplot.title(title)
    pyplot.pie(freqs, labels=vals, autopct='%1.1f%%')
    
    pyplot.savefig(fileName)
    pyplot.close()

def makeDotChart(vals, title, labelX, fileName):
    """
    Param1: A list of  values
    Param2: A string for the graph title
    Param3: A string for the x axis label
    Saves a PDF of a Dot Chart
    Description: Vals is just a raw list of numbers (such as an entire attribute column)
    """
    vals.sort()
    ys = [1] * len(vals)
    pyplot.figure()
    pyplot.title(title)
    pyplot.plot(vals, ys, 'b.', alpha=0.2, markersize=16)
    pyplot.gca().get_yaxis().set_visible(False)
    
    pyplot.savefig(fileName)
    pyplot.close()

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
    
def getBinsFromValues(vals, cutOffs):
    """
    Param1: A list of numeric values
    Param2: A list of numeric cutOff points to be converted into intervals
    and cutOffs(a list of cutOff points)
    Returns a list of equal length to vals 
    Each value in the returned list specifies which bin the corresponding
    vals value at the same index should fall in.
    """
    bins = []
    for item in vals:
#        if(item > cutOffs[len(cutOffs)-1]):
#            bins.append(len(cutOffs))
#        else:
        for i in range(len(cutOffs)):
            if item <= cutOffs[i]:
                bins.append(i+1)
                break
            if i == len(cutOffs) - 1:
                bins.append(i+2)
                break
    return bins

def getXLabelsFromCutOffs(cutOffs):
    """"
    Param1: A list of cut off point numeric values
    Returns a list of strings that are the labels for the for bins
    which specify the interval ranges for a histogram.
    length of returned list is +1 greater than the length of the cutOffs
    """
    labels = []
    labels.append("<=" + str(cutOffs[0]))
    for i in range(len(cutOffs)-1):
        labels.append(str(cutOffs[i]+1)+"--"+str(cutOffs[i+1]))
    labels.append(str(cutOffs[len(cutOffs)-1]+1)+"<=")
    return labels
    
def makeScatterPlot(xVals, yVals, title, labelX, labelY, includeLinRegLine, fileName):
    """
    Param1: A list of numeric X values
    Param2: A list of numeric Y values
    Param3: A string for the graph title
    Param4: A string for the x axis label
    Param5: A string for the y axis label
    Param6: A boolean value for whether to include Linear Regression line
    Returns: A saved PDF of the Scatter Plot
    """
    pyplot.figure()
    pyplot.title(title)
    pyplot.xlabel(labelX)
    pyplot.ylabel(labelY)
    pyplot.plot(xVals, yVals, 'b.') 
    minXVal = min(xVals)
    maxXVal = max(xVals)
    minYVal = min(yVals)
    maxYVal = max(yVals)
    xRange = maxXVal - minXVal
    yRange = maxYVal - minYVal
    pyplot.xlim(minXVal - .1*xRange, maxXVal + .1*xRange)
    pyplot.ylim(minYVal - .1*yRange, maxYVal + .1*yRange)
    if(includeLinRegLine):
        avgX = calcMeanValue(xVals)
        avgY = calcMeanValue(yVals)
        slope = findSlope(xVals,yVals)
        yIntercept = findIntercept(slope, avgX, avgY)
        x = numpy.arange(0, max(xVals), .1)
        y = slope * x + yIntercept
        pyplot.plot(x, y)
        r = findCorrelationCoeff(xVals, yVals, avgX, avgY)
        pyplot.text(minXVal - .1*xRange, maxYVal + .2*yRange, "Correlation Coefficient: "+str(r), bbox=dict(facecolor='red', alpha=0.5))
    
    pyplot.savefig(fileName)
    pyplot.close()
   
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

def makeBoxPlot(grouping, XLabelValues, title, labelX, labelY, fileName):
    """
    Param1: A List of List (of values) which is the data for 
    each side by side box plot
    Param2: The x label values for each boxplot (Same length as grouping)
    Param3: A title string
    Param4: A x label string
    Param5: A y label string
    """
    pyplot.figure()
    pyplot.title(title)
    pyplot.xlabel(labelX)
    pyplot.ylabel(labelY)
    pyplot.boxplot(grouping)
    pyplot.xticks(range(1, len(grouping)+1),XLabelValues)
    pyplot.savefig(fileName)
    pyplot.close()

def doQuestion8Part2(table, origin, modelYear):
    listOfYears = getAttrColumn(table, modelYear)
    listOfOrigins = getAttrColumn(table, origin)
    uniqueListOfOrigins = list(set(listOfOrigins)) #Removes Dups.. quickly
    uniqueListOfOrigins.sort()
    uniqueListOfYears = list(set(listOfYears))
    uniqueListOfOrigins.sort()
    carCountsOriginByYear = []
    for i in uniqueListOfOrigins:
        carCountsOriginByYear.append([])
    for i in uniqueListOfYears:
        for j in range(len(carCountsOriginByYear)):
            carCountsOriginByYear[j].append(0)
    for item in table:
        originVal = item[origin]
        yearVal = item[modelYear]
        indexOfYear = uniqueListOfYears.index(yearVal)
        indexOfOrigin = uniqueListOfOrigins.index(originVal)
        carCountsOriginByYear[indexOfOrigin][indexOfYear] = carCountsOriginByYear[indexOfOrigin][indexOfYear] + 1
        
    pyplot.figure()
    fig, ax = pyplot.subplots()
    positionList = [0,1,2,3,4,5,6,7,8,9]
    colorList = ['r','b','g','yellow','purple','orange','pink','brown']
    returns = []
    for i in range(len(carCountsOriginByYear)):
        returns.append([])
        returns[i] = ax.bar(positionList, carCountsOriginByYear[i], 0.3, color=colorList[i])[0]
        positionList = [x+.3 for x in positionList]
    xticks = numpy.arange(.5, len(uniqueListOfYears)+1)
    ax.set_xticks(xticks)
    ax.set_yticks([10,20,30,40,50])
    ax.set_xticklabels([str(x) for x in uniqueListOfYears])
    ax.legend(tuple(returns), ("US", "Europe", "Japan"), loc=2)
    pyplot.ylabel("Number of Cars")
    pyplot.xlabel("Year")
    pyplot.grid(True)
    pyplot.savefig("step-8-number-cars.pdf")
    pyplot.close()


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
    mpgList = getAttrColumn(table, mpg)
    cylindersList = getAttrColumn(table, cylinders)
    displacementList = getAttrColumn(table, displacement)
    horsepowerList = getAttrColumn(table, horsepower)
    weightList = getAttrColumn(table, weight)
    accelerationList = getAttrColumn(table, acceleration)
    modelYearList = getAttrColumn(table, modelYear)
    originList = getAttrColumn(table, origin)
    carNameList = getAttrColumn(table, carName)
    msrpList = getAttrColumn(table, msrp)
    
    #Part 1: Frequency Diagram of Catagories
    makeHistogram(getAttrColumn(table, cylinders), [],"Frequencies of Cylinders (Historgram)", "Number of Cylinders", "Frequencies", "step-1-clyinders.pdf")
    makeHistogram(getAttrColumn(table, modelYear), [],"Frequencies of Model Year (Historgram)", "Model Year", "Frequencies", "step-1-model-year.pdf")
    makeHistogram(getAttrColumn(table, origin), [],"Frequencies of Origins (Historgram)", "Origin", "Frequencies", "step-1-origins.pdf")
    
    #Part 2: Pi Chart of Catagories
    makePieChart(getAttrColumn(table, cylinders), "Frequencies of Cylinders (Pie Chart)", "step-2-clyinders.pdf")
    makePieChart(getAttrColumn(table, modelYear), "Frequencies of Model Year (Pie Chart)", "step-2-model-years.pdf")
    makePieChart(getAttrColumn(table, origin), "Frequencies of Origins (Pie Chart)", "step-2-origins.pdf")
    
    #Part 3: Dot Plot for Continuous attributes
    makeDotChart(getAttrColumn(table, mpg), "Frequencies of MPG (Dot Chart)", "MPG", "step-3-mpg.pdf")
    makeDotChart(getAttrColumn(table, displacement), "Frequencies of Displacement (Dot Chart)", "Displacement", "step-3-displacement.pdf")
    makeDotChart(getAttrColumn(table, horsepower), "Frequencies of Horsepower (Dot Chart)", "Horsepower", "step-3-horsepower.pdf")
    makeDotChart(getAttrColumn(table, weight), "Frequencies of Weight (Dot Chart)", "Weight", "step-3-weight.pdf")
    makeDotChart(getAttrColumn(table, acceleration), "Frequencies of Acceleration (Dot Chart)", "Acceleration", "step-3-acceleration.pdf")
    makeDotChart(getAttrColumn(table, msrp), "Frequencies of MSRP (Dot Chart)", "MSRP", "step-3-msrp.pdf")
    
    #Part 4: Approach 1: Frequency Diagram of MPG Ratings
    ratings = getRatingsForMPG(mpgList)
    xLabelValues = ["<=13","14","15--16","17--19","20--23","24--26","27--30","31--36","37--44",">=45"]
    makeHistogram(ratings, xLabelValues, "Frequencies of MPG (Rating Bins)", "Rating", "Frequencies", "step-4-mpg-manual-cutoffs.pdf")
    
    #Part 4: Approach 2: Frequency Diagram of MPG Ratings
    cutOffs = getEqualWidthBins(mpgList, 5)
    bins = getBinsFromValues(mpgList, cutOffs)
    xLabelValues = getXLabelsFromCutOffs(cutOffs)
    makeHistogram(bins, xLabelValues, "Frequencies of MPG (5 Bins)", "MPG Bins", "Frequencies", "step-4-mpg-5bins.pdf")
    
    #Part 5: Frequency Histrograms for each Continuous attribute 
    #mpg
    cutOffs = getEqualWidthBins(mpgList, 10)
    bins = getBinsFromValues(mpgList, cutOffs)
    xLabelValues = getXLabelsFromCutOffs(cutOffs)
    makeHistogram(bins, xLabelValues, "Frequencies of MPG (10 Bins)", "MPG Bins", "Frequencies", "step-5-mpg.pdf")
    #displacement
    cutOffs = getEqualWidthBins(displacementList, 10)
    bins = getBinsFromValues(displacementList, cutOffs)
    xLabelValues = getXLabelsFromCutOffs(cutOffs)
    makeHistogram(bins, xLabelValues, "Frequencies of Displacement (10 Bins)", "Displacement Bins", "Frequencies", "step-5-displacement.pdf")
    #horsepower
    cutOffs = getEqualWidthBins(horsepowerList, 10)
    bins = getBinsFromValues(horsepowerList, cutOffs)
    xLabelValues = getXLabelsFromCutOffs(cutOffs)
    makeHistogram(bins, xLabelValues, "Frequencies of Horsepower (10 Bins)", "Horsepower Bins", "Frequencies", "step-5-horsepower.pdf")
    #weight
    cutOffs = getEqualWidthBins(weightList, 10)
    bins = getBinsFromValues(weightList, cutOffs)
    xLabelValues = getXLabelsFromCutOffs(cutOffs)
    makeHistogram(bins, xLabelValues, "Frequencies of Weight (10 Bins)", "Weight Bins", "Frequencies", "step-5-weight.pdf")
    #acceleration
    cutOffs = getEqualWidthBins(accelerationList, 10)
    bins = getBinsFromValues(accelerationList, cutOffs)
    xLabelValues = getXLabelsFromCutOffs(cutOffs)
    makeHistogram(bins, xLabelValues, "Frequencies of Acceleration (10 Bins)", "Acceleration Bins", "Frequencies", "step-5-acceleration.pdf")
    #msrp
    cutOffs = getEqualWidthBins(msrpList, 10)
    bins = getBinsFromValues(msrpList, cutOffs)
    xLabelValues = getXLabelsFromCutOffs(cutOffs)
    makeHistogram(bins, xLabelValues, "Frequencies of MSRP (10 Bins)", "MSRP Bins", "Frequencies", "step-5-msrp.pdf")
#    
#    #Part 6: Scatter Plots for continuous attributes
    makeScatterPlot(displacementList, mpgList, "Displacement vs MPG", "Displacement", "MPG", False, "step-6-displacement.pdf")
    makeScatterPlot(horsepowerList, mpgList, "Horsepower vs MPG", "Horsepower", "MPG", False, "step-6-horsepower.pdf")
    makeScatterPlot(weightList, mpgList, "Weight vs MPG", "Weight", "MPG", False, "step-6-weight.pdf")
    makeScatterPlot(accelerationList, mpgList, "Acceleration vs MPG", "Acceleration", "MPG", False, "step-6-acceleration.pdf")
    makeScatterPlot(msrpList, mpgList, "MSRP vs MPG", "MSRP", "MPG", False, "step-6-msrp.pdf")
    
    #Part 7: Scatter Plot Linear Regression lines
    makeScatterPlot(displacementList, mpgList, "Displacement vs MPG (with Linear Regression)", "Displacement", "MPG", True, "step-7-displacement.pdf")
    makeScatterPlot(horsepowerList, mpgList, "Horsepower vs MPG (with Linear Regression)", "Horsepower", "MPG", True, "step-7-horsepower.pdf")
    makeScatterPlot(weightList, mpgList, "Weight vs MPG (with Linear Regression)", "Weight", "MPG", True, "step-7-weight.pdf")
    makeScatterPlot(msrpList, mpgList, "MSRP vs MPG (with Linear Regression)", "MSRP", "MPG", True, "step-7-msrp.pdf")
    
    #Part 8: Two graphs:
    
    #8Part 1: Box Plot of MPG (continuous) by year (categorical). 
    rowGroupedByYear = groupBy(table, modelYear)
    mpgGroupedByYear = rowGroupedByYear
    listOfYears = []
    for i in range(len(mpgGroupedByYear)):
        listOfYears.append(mpgGroupedByYear[i][0][modelYear])
        for j in range(len(mpgGroupedByYear[i])):
            mpgGroupedByYear[i][j] = mpgGroupedByYear[i][j][mpg]
    makeBoxPlot(mpgGroupedByYear, listOfYears, "MPG by Model Year", "Years", "MPG", "step-8-boxplot.pdf")
    
    #8Part 2:  Frequency Diagram of cars from each country (categorical) separated
    #out by model year (cataegorical)
    doQuestion8Part2(table, origin, modelYear)
    
main()

if __name__ == '__main__':
    main()
