"""
   This program finds the attractors of an iterated function system
   Dr. Y    April 10, 2014
"""
import random
import math
import turtle

def getBeta():
    beta = float(input("Enter beta in open (0, 1) -> "))
    return beta

def getNumberOfFunctions():
    number_Functions = int(input("Enter number of fixed points -> "))
    return number_Functions

def getRotationAngle():
    theta = float(input("Enter nonzero rotation angle in radians -> "))
    return theta

def getListOfPoints(number):
   
    pointsList = []
    for i in range(number):
        point = []
        point_x = float(input("Enter x-coordinate of point -> "))
        point_y = float(input("Enter y-coordinate of point -> "))
        point.append(point_x)
        point.append(point_y)
        pointsList.append(point)
    return pointsList

def getRandomPointInPointsList(pen, number, pointsList):
    whichOne = int(number * random.random())
    setPenColor(pen, whichOne, pointsList);
    return pointsList[whichOne]

def doFunctionIterate(beta, theta, point, fixed_point):
     new_x = math.cos(theta) * (point[0] - fixed_point[0])
     new_x = new_x - math.sin(theta) * (point[1] - fixed_point[1])
     new_x = beta * new_x + fixed_point[0]
     new_y = math.sin(theta) * (point[0] - fixed_point[0])
     new_y = new_y + math.cos(theta) * (point[1] - fixed_point[1])
     new_y = beta * new_y + fixed_point[1]
     new_point = []
     new_point.append(new_x)
     new_point.append(new_y)
     return new_point
    
def eventualOrbit(pen, beta, theta, number, pointsList, numberIterates):
    plotPoint = getRandomPointInPointsList(pen, number, pointsList)
    for i in range(numberIterates):
        fixed_point = getRandomPointInPointsList(pen, number, pointsList)
        plotPoint = doFunctionIterate(beta, theta, plotPoint, fixed_point)
        pen.up()
        pen.goto(plotPoint)
        pen.down()
        pen.dot(3)

def setPenColor(pen, whichOne, pointList):
    if(whichOne == 0):
        pen.color("blue")
    elif(whichOne == 1):
        pen.color("red")
    elif(whichOne == 2):
        pen.color("yellow")
    elif(whichOne == 3):
        pen.color("green")
    elif(whichOne == 4):
        pen.color("black")
    elif(whichOne == 5):
        pen.color("purple")
    elif(whichOne == 6):
        pen.color("brown")
    elif(whichOne == 7):
        pen.color("orange")
    elif(whichOne == 8):
        pen.color("pink")
    elif(whichOne == 9):
        pen.color("grey")
        

def setupScreen (screen, bottom_x, bottom_y, top_x, top_y):
    screen.setworldcoordinates(bottom_x, bottom_y, top_x, top_y)
    screen.tracer(20000)
    
def setupPen(pen):
    pen.hideturtle()
    pen.color("blue")
    
    
def main():
    screen = turtle.Screen()
    pen = turtle.Turtle()
    setupScreen(screen, -1,-1, 3,3)
    setupPen(pen)

    beta = getBeta()
    number_Functions = getNumberOfFunctions()
    theta = getRotationAngle()
    pointsList = getListOfPoints(number_Functions)
   
    eventualOrbit(pen, beta, theta, number_Functions, pointsList, 80000)
 
    screen.exitonclick()
main()        
        
        
        
