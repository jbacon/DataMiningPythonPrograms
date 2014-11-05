#Gisela Arreola-Gutierrez
#Josh Bacon
#Lab 2 Part 2
#log1.py

import math
import turtle
import sys

def Q(c, x):
    return c* x* (1 - x)
    
#horizontal c-axis from 1 to 4
#vertical x-axis from 0 to 1
def drawAxis(pen):
    pen.up()
    pen.goto(-0.01, 0)
    pen.write(0, False, align="right")
    pen.down()
    pen.goto(0.01, 0)
    pen.goto(0, 0)
    pen.goto(0, 1)
    pen.goto(-0.01, 1)
    pen.write(1, False, align="right")
    pen.goto(0.01, 1)
    pen.write("  x", False, align="left")
    pen.up()
    pen.goto(0, 0)
    pen.down()
    pen.goto(4, 0)
    pen.write("c", False, align="left")
    pen.up()
    
#puts a vertical mark on the c-axis at value c and writes the value of c below the axis
def markAxis(pen, c):
    pen.up()
    pen.goto(c, 0.05)
    pen.down()
    pen.goto(c, -0.05)
    pen.up()
    pen.goto(c, -0.1)
    pen.write(round(c, 2), False, align="center")
    pen.up()
    
#using a seed of 0
#itereate 100 times to get to the limiting behavior
#plot the next 50 iterates - dot size 3 seems good
def eventualOrbit(pen, c):
    orbitNum = .5
    for i in range(100):
        orbitNum = Q(c, orbitNum)
    for i in range(50):
        orbitNum = Q(c, orbitNum)
        pen.up()
        pen.goto(c, orbitNum)
        pen.dot()
        pen.up()

#[1, 3] : Exactly 10 c values, including 1 & 3
def drawFirstInterval(pen) :
    pen.color("red")
    for i in range(0, 11) :
        c = 1 + i/5  
        eventualOrbit(pen, c)
    markAxis(pen, c)
    
#(3, 3.45] : Exactly 9 c values, including only 3.45
def drawSecondInterval(pen) :
    pen.color("blue")
    for i in range(1, 10) :
        c = 3 + ((i/10) * 0.5)
        eventualOrbit(pen, c)
    markAxis(pen, c)

#(3.45, 3.54] : Exactly 13 cvalues, including 3.54
def drawThirdInterval(pen) :
    pen.color("green")
    for i in range (1, 14) :    
        c = 3.45 + ((i/21) * 0.15)
        eventualOrbit(pen, c)
    markAxis(pen, c)

#(3.54, 4.0] : Exactly 10 c values, including only 3.54
def drawFourthInterval(pen) :
    pen.color("red")
    for i in range (0, 11) :
        c = 3.54 +((i/10) * 0.46)    
        eventualOrbit(pen, c)
    markAxis(pen, c)

def main():
    screen = turtle.Screen()
    screen.setworldcoordinates(-.1,-.1, 4.1, 1.1)
    screen.tracer(1000)
    pen = turtle.Turtle()
    pen.shape("circle")
    pen.pensize(2)
    pen.hideturtle()
    drawAxis(pen)
    c = 1
    markAxis(pen, c)
    drawFirstInterval(pen)
    drawSecondInterval(pen)
    drawThirdInterval(pen)
    drawFourthInterval(pen)
    screen.exitonclick()
main()
