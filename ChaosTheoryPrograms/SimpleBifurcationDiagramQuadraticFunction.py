#Josh Bacon
#Gisela
#quadc1.py

import math
import turtle
import sys

#The Quadratic Function
def Q(c, x):
    return (x * x + c)

#Creates the Axis for the graph
#horizontal c-axis from -1.75 to 0.25
#vertical x-axis from -3 to 3
def drawAxis(pen):
    pen.up()
    pen.goto(-0.01, -3)
    pen.write(-3, False, align="right")
    pen.down()
    pen.goto(0.01, -3)
    pen.goto(0, -3)
    pen.goto(0, 3)
    pen.goto(-0.01, 3)
    pen.write(3, False, align="right")
    pen.goto(0.01, 3)
    pen.write("  x", False, align="left")
    pen.up()
    pen.goto(-1.75, 0)
    pen.down()
    pen.goto(.25, 0)
    pen.write("c", False, align="left")
    pen.up()

#Marks the c-axis with vertical mark and writes value
def markAxis(pen, c):
    pen.up()
    pen.goto(c, 0.1)
    pen.down()
    pen.goto(c, -0.1)
    pen.up()
    pen.goto(c, -0.2)
    pen.write(round(c, 2), False, align="center")
    pen.up()

#Calculates and plots on the graph the eventual Orbit of 0
#using a seed of 0
#itereate 100 times to get to the limiting behavior
#plot the next 50 iterates - dot size 3 seems good
def eventualOrbit(pen, c):
    orbitNum = 0
    for i in range(100):
        orbitNum = Q(c, orbitNum)
    for i in range(50):
        orbitNum = Q(c, orbitNum)
        pen.up()
        pen.goto(c, orbitNum)
        pen.dot()
        pen.up()

#Draws the first interval on the graph
#[-0.75, 0.25] : Exactly 10 c values, including -.75 & .25
def drawFirstInterval(pen) :
    c = 0.25
    pen.color("red")
    markAxis(pen, c)
    for i in range(0, 10) :
        c = 0.25 - i/9  #Is i/9, and not i/10 because interval is inclusive of both endpoints instead of just 1.
        eventualOrbit(pen, c)
    markAxis(pen, c)

#Draws the second interval on the graph
#[-1.25, -0.75) : Exactly 10 c values, including only -1.25
def drawSecondInterval(pen) :
    pen.color("blue")
    for i in range(1, 11) :
        c = -0.75 - ((i/10) * 0.5)
        eventualOrbit(pen, c)
    markAxis(pen, c) 

#Draw the third interval on the graph
#[-1.4, -1.25) : Exactly 21 c values, including only -1.4. Makes the graph look like a line
def drawThirdInterval(pen) :
    pen.color("green")
    for i in range (1, 22) :    
        c = -1.25 - ((i/21) * 0.15)
        eventualOrbit(pen, c)
    markAxis(pen, c)

#Draws the fourth interval on the graph
#[-1.75, -1.4) : Exactly 6 c values, including only -1.75
def drawFourthInterval(pen) :
    pen.color("red")
    
    for i in range (1, 7) :
        c = -1.4 - ((i/6) * 0.35)    
        eventualOrbit(pen, c)
    markAxis(pen, c)

def main():
    screen = turtle.Screen()
    screen.setworldcoordinates(-1.8,-3, .3, 3)
    screen.tracer(1000)
    pen = turtle.Turtle()
    pen.shape("circle")
    pen.pensize(3)
    pen.hideturtle()
    drawAxis(pen)
    c = 0.25
    pen.color("red")
    markAxis(pen, c)
    drawFirstInterval(pen)
    drawSecondInterval(pen)
    drawThirdInterval(pen)
    drawFourthInterval(pen)
    screen.exitonclick()
main()
