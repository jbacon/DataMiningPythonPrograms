"""@chaos game: starting with 3 vertices of an equilateral triangle and
   the choice of "the point" in the plane, choose one of the vertices at random,
   the next iterate is the midpoint between "the point" and the chosen
   vertex. the midpoint then becomes "the point" and iterations continue
   @authors   chaos class on
   @date January 15, 2014
   @file chaosgame.py
"""
import random
import math
import turtle

def FindMidpoint(pointP, pointQ):
    midpoint = []
    for i in range(2):
        midpoint.append((pointP[i] + pointQ[i]) / 2)
    return midpoint

def GetRandomVertex(Vertices):
    num = int(3 * random.random())
    return Vertices[num]

def GetRandomPoint():
    x = 2 * random.random()
    y = 2 * random.random()
    point = list()
    point.append(x)
    point.append(y)
    return point

def DoIterations(pen, point, Vertices, numberOfTimes):
    for i in range(numberOfTimes):
        vertex = GetRandomVertex(Vertices)
        point = FindMidpoint(point, vertex)
        pen.up()
        pen.goto(point)
        pen.down()
        pen.dot(3)
        
def main():
    screen = turtle.Screen()
    screen.setworldcoordinates(0,0, 2,2)
    screen.tracer(1000)
    
    pen = turtle.Turtle()
    pen.hideturtle()
    pen.color("blue")

    Vertices = [[0,0], [2,0], [1, math.sqrt(3)]]
    point = GetRandomPoint()
    DoIterations(pen, point, Vertices, 10000)
    screen.exitonclick()
    
main()        
        
        
        
