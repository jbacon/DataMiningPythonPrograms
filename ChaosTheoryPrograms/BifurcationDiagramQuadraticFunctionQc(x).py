"""
@Author: Cj Buresch, Josh Bacon 
@Date: 3-22-2014
@Assignment: Lab #3
@Description: Finds the ultimate fate of the orbit 0, then
    plots the next 50 iterations. The pliot is a histogram of
    the functions cycle.Now includes user input to "zoom" in on
    sections of the graph
@File: lab3.py
@Python Version: 3.3.4
"""

import math
import turtle
import sys
import random

#Draw the vertical and horizontal axis on the screen
#Will draw at zero if the window is started at greater than
#zero, otherwise will draw a vertical axis specific to the
#window location
def draw_Axes(clist,xlist,left,right,bottom,top,vstretch,pen):
    #Where Axis are drawn
    xaxisLocation = 0
    caxisLocation = 0
    if(right < 0):
        xaxisLocation = right
    elif(left > 0):
        xaxisLocation = left
    if(top < 0):
        caxisLocation = top
    elif(bottom > 0):
        caxisLocation = bottom

    #Draws the C axis
    penmovetool(left, caxisLocation, pen)
    pen.goto(right, caxisLocation)
    #Draws the X axis
    penmovetool(xaxisLocation, vstretch * top, pen)
    pen.goto(xaxisLocation, vstretch * bottom)
    #Marks Axis
    markCAxis_withList(pen, caxisLocation,bottom,top,clist)
    markXAxis_withList(pen, xaxisLocation,left,right,xlist,vstretch)

#use a list to draw c-axis markings on the screen   
def markCAxis_withList(pen,caxisLocation,bottom,top,clist):
    offset = abs(top - bottom)/100
    for c in clist:
        mark_CAxis(pen,offset, caxisLocation, c)
        
#use a list to draw x-axis markings on the screen
def markXAxis_withList(pen,xaxisLocation,left,right,xlist,vstretch):
    offset = abs(right - left)/100
    for x in xlist:
        mark_XAxis(pen,offset,xaxisLocation,x,vstretch)

#create a c-axis tick at c on the screen 
def mark_CAxis(pen,offset,caxisLocation,c):
    penmovetool(c,caxisLocation + offset,pen)
    pen.write("%.3f" % c)
    penmovetool(c,caxisLocation - offset,pen)
    pen.goto(c,caxisLocation + offset)

#create a x-axis tick at "right" on thescreen
def mark_XAxis(pen,offset,xaxisLocation,x,vstretch):
    penmovetool(xaxisLocation - (4 * offset),x * vstretch,pen)
    pen.write("%.3f" % x)
    penmovetool(xaxisLocation - offset,x * vstretch,pen)
    pen.goto(xaxisLocation + offset,x * vstretch)

#draws each orbit in a list on the screen, changes the color
#after drawing a defined number of orbits
def listtoOrbit(pen,clist,iterations,vstretch):
    for c in clist:
        readyToChange(c,pen)
        eventual_Orbit(pen, c,iterations,vstretch)
            
#Change the Color
def readyToChange(c,pen):
    if(c > -.75):
        #1-cycle
        pen.color("red")
    elif(c <= -.75 and c > -1.25):
        #2-cycle
        pen.color("blue")
    elif(c <= -1.25 and c > -1.474):
        #4-cyle
        pen.color("green")
    elif(c <= -1.474 and c > -1.522):
        #6-cyle
        pen.color("orange")
    elif(c <= -1.522 and c > -1.575):
        #8-cyle
        pen.color("purple")
    elif(c <= -1.575 and c > -1.626):
        #7-cyle
        pen.color("blue")
    elif(c <= -1.626 and c > -1.674):
        #5-cyle
        pen.color("green")
    elif(c <= -1.674 and c > -1.711):
        #7-cyle
        pen.color("red")
    elif(c <= -1.711 and c > -1.76):
        #8-cyle
        pen.color("orange")
    elif(c <= -1.76 and c > -2):
        #3-cyle
        pen.color("green")
#runs the seed through 100 iterations, then plots the
#next defined number of iterations. Now includes a
# vertical stretch option.
def eventual_Orbit(pen,c,iterations,vstretch):
    i = 0;
    seed = 0;
    for i in range(0,100):
        seed = Q(c,seed)
    for j in range(0,iterations):
        temp = Q(c,seed)
        penmovetool(c,temp * vstretch,pen)
        pen.dot(3)
        seed = temp

#Handy tool that saves me from writing this all the time
def penmovetool(x,y,pen):
    pen.up()
    pen.goto(x,y)
    pen.down()

#The Function used in this experiment
def Q(c,x):
    return (x * x) + c

#generates a list of values from a lower negative value
#to a higher negative value, step by step. This creates
#everything from tick marks to the values of c 
def generateNegList(start,stop,step):
    alist = []
    while start > stop:
        alist.append(start + step)
        start += step
    return alist

#used to create automatic ticks instead of the previous
#hardcoded axis marks
def genAxisList(num,left,right):
    step = math.fabs(left - right) / num
    return generateNegList(right,left,-step)

def main():
    #settings for this run
    num_iterations = 50
    #get user input
    cleft = float(input("Minimum value for c: "))
    cright = float(input("Maximum value for c: "))
    num_cmarks = int(input("c value tick marks: "))

    xmin = float(input("Minimum value for x: "))
    xmax = float(input("Maximum value for x: "))
    vstretch = float(input("Vertical Stretch: "))
    xmin_stretch = xmin * vstretch
    xmax_stretch = xmax * vstretch
    detail = (-1*abs(cright - cleft))/num_cmarks
    #generate screen with input and initialize turtle
    cMargin = abs(cright - cleft)/50
    xMargin = abs(xmin - xmax)/50
    screen = turtle.Screen()
    screen.tracer(1000)
    screen.setworldcoordinates(cleft - cMargin,xmin - xMargin,cright + cMargin,xmax + xMargin)
    screen.bgcolor("white")
    
    pen = turtle.Turtle()
    pen.hideturtle()
    pen.color("black")
    
    #generate lists for axis and draw on screen
    if(num_cmarks < 20):
        Caxislist = genAxisList(num_cmarks,cleft,cright)
    else:
        Caxislist = genAxisList(20,cleft,cright)
        
    Xaxislist = genAxisList(10,xmin,xmax)
    
    draw_Axes(Caxislist,Xaxislist,cleft,cright,xmin,xmax,vstretch,pen)

    #Generate orbit list for orbits and run
    pen.color("red")
    clist1 = generateNegList(cright,cleft,detail)
    listtoOrbit(pen,clist1,num_iterations,vstretch)

    screen.exitonclick()
    
main()
