""" Josh Bacon, Evan Shioyama
    Filename: graphs.py
    Date: Jan 27th, 2014
    Description: Program gives user a menu to choose which graph the turtle should draw,
        Graph a is the function y = cos(x) - x
        Graph b requires the user to enter a value float number to graph y = num * x * (1-x)
    Inputs: Menu selection and float number
    Outputs: Graph on screen
"""

import math
import turtle
import sys

#Comment
def createAxis() :
    turtle.up()
    turtle.goto(-5, 0)
    turtle.down()
    turtle.goto(5, 0)
    turtle.up()
    turtle.goto(0, 5)
    turtle.down()
    turtle.goto(0, -5)
    turtle.up()

#Comment
def functionA(x) :
    y = math.cos(x) - x
    return y

#Comment
def functionB(x, userFloat) :
    y = userFloat * x * (1 - x)
    return y
""" You can remove this stuff, I dont think we need it

def derivA(x) :
    y = -1 * math.sin(x) - 1
    return y

def derivB(x) :
    y = userFloat - 2 * userFloat * x
    return y

def tanLineFunctionA(xFunction, xTanLine) :
    global userFloat
    y = functionA(xFunction)                        #Gets y from Function a, used to find Tangent Line Eq
    slope = derivA(xFunction)                       #Gets slope at xFunction point on Function a
    b = y - slope * xFunction                       #Gets b for tangent line equation
    yTanLine = slope * xTanLine + b                 #Gets yPoint from tangent line equation
    return yTanLine

def tanLineFunctionB(xFunction, xTanLine) :
    global userFloat
    y = functionB(xFunction)                        #Gets y from Function a, used to find Tangent Line Eq
    slope = derivB(xFunction)                       #Gets slope at xFunction point on Function a
    b = y - slope * xFunction                       #Gets b for tangent line equation
    yTanLine = slope * xTanLine + b                 #Gets yPoint from tangent line equation
    return yTanLine

def drawTanLineA(pen, x) :
    pen.up()
    pen.goto(0, tanLineFunctionA(x, 0))
    pen.down()
    pen.goto(2, tanLineFunctionA(x, 2))
    pen.up()

def drawTanLineB(pen, x) :
    pen.up()
    pen.goto(0, tanLineFunctionB(x, 0))
    pen.down()
    pen.goto(3, tanLineFunctionB(x, 3))
"""
#comment
def graphFunctionA() :
    x = 0
    turtle.color("red")
    turtle.up()
    y = functionA(x)
    turtle.goto(x, y)
    turtle.down()
    while(x <= 2) :
        y = functionA(x)
        turtle.goto(x, y)
        x = x + .1
    
#comment
def graphFunctionB(userFloat) :
    x = 0
    turtle.up()
    while(x <= 3) :
        y = functionB(x, userFloat)
        turtle.goto(x, y)
        turtle.dot(10, "red")
        x = x + .2
#Comment
def main () :
    menuOption = input("a) y = cos(x) - x on [0,2] as a continous curve \nb) y = userFloat * (1 - x) on [0,3] as data points only \nx) exit program\n")
    while(True) :
        if menuOption == "a" :
            screen = turtle.Screen()
            screen.setworldcoordinates(-5, -5, 5, 5)
            screen.tracer(1000)
            graphFunctionA()#Note: For some reason if I reverse the order of 
            createAxis()    #graphFunctionA() and createAxis() the turtle stops drawing on some parts (no idea why)
            break
        elif menuOption == "b" :
            try :
                userFloat = float(input("Enter a float value: "))
            except :
                #Something
                sys.exit()
            screen = turtle.Screen()
            screen.setworldcoordinates(-5, -5, 5, 5)
            screen.tracer(1000)
            graphFunctionB(userFloat) 
            createAxis()
            break
        elif menuOption == "x" :
            sys.exit(0);
        else :
            menuOption = input("That is an invalid option, please enter a correct option:")
    screen.exitonclick()
main()

    
    
