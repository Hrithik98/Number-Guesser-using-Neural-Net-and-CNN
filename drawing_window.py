import keras 
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import tensorflow as tf
import pygame
import numpy as np
from tkinter import *
from tkinter import messagebox
import matplotlib.pyplot as plt

print(tf.__version__," ",keras.__version__)

class pixels(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = (255,255,255)
        self.neighbours = []

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.x+self.width, self.y+self.height))

    def getNeighbours(self, grid):
        j = self.x // 20
        i = self.y // 20
        rows = cols = 28

        if i < cols-1 : self.neighbours.append(grid.pixels[i+1][j])
        if i > 0 : self.neighbours.append(grid.pixels[i-1][j])
        if j < rows-1 : self.neighbours.append(grid.pixels[i][j+1])
        if j > 0 : self.neighbours.append(grid.pixels[i][j-1])

        if i > 0 and j > 0 :self.neighbours.append(grid.pixels[i-1][j-1])
        if j < rows-1 and i-1 >= 0 : self.neighbours.append(grid.pixels[i-1][j+1])
        if i < cols-1 and j < rows-1 : self.neighbours.append(grid.pixels[i+1][j+1])
        if i < cols-1 and j-1 >= 0 : self.neighbours.append(grid.pixels[i+1][j-1])

class grid(object):
    pixels = []
    def __init__(self, row, col, width, height):
        self.rows = row
        self.cols = col
        self.len = row * col
        self.width = width
        self.height = height
        self.generatePixels()

    def draw(self, surface):
        for row in self.pixels:
            for col in row:
                col.draw(surface)

    def generatePixels(self):
        pixel_width = self.width // self.cols
        pixel_height = self.height // self.rows
        self.pixels = []
        for r in range(self.rows):
            temp = []
            for c in range(self.cols):
                temp.append(pixels(pixel_width * c, pixel_height * r, pixel_width, pixel_height))
            self.pixels.append(temp)

        for r in range(self.rows):
            for c in range(self.cols):
                self.pixels[r][c].getNeighbours(self)

    def clicked(self, pos):
        t = pos[0]
        w = pos[1]
        g1 = int(t)//self.pixels[0][0].width
        g2 = int(w)//self.pixels[0][0].height
        return self.pixels[g2][g1]

    def convert_binary(self):
        newMatrix = []
        for i in range(len(self.pixels)):
            temp = []
            for j in range(len(self.pixels[i])):
                if self.pixels[i][j].color == (255,255,255): temp.append(0)
                else : temp.append(1)
            newMatrix.append(temp)

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = tf.keras.utils.normalize(x_test, axis=1)
        for row in range(28):
            for x in range(28):
                x_test[0][row][x] = newMatrix[row][x]

        return x_test[:1]

def guess(li):
    img = li
    print(li.shape)
    image = img.reshape((28,28,1))
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model1 = load_model('model.h5', compile=False)
        model2 = load_model('m.model', compile=False)
    ##model.summary()
    pred1 = model1.predict(image)
    pred2 = model2.predict(li)
   # print(pred1[0], pred2[0])
    t1 = np.argmax(pred1[0])
    t2 = np.argmax(pred2[0])
    print("Predicted number : ")
    print("By a Neural Network : ", t2)
    print("By a Convolutional Neural Network : ", t1)
    window = Tk()
    window.withdraw()
    messagebox.showinfo("Predictions", "The number predicted by : \n"+ "Convolutional Neural Network is "+str(t1)+" and \n"+"Neural Network is "+str(t2))
    window.destroy()


def main():
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                li = grid.convert_binary()
                guess(li)
                grid.generatePixels()

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                clicked = grid.clicked(pos)
                clicked.color = (0,0,0)

                for n in clicked.neighbours : n.color = (0,0,0)

            if pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                clicked = grid.clicked(pos)
                clicked.color = (255,255,255)

        grid.draw(display_game)
        pygame.display.update()
    

pygame.init()
width = 560
height = 560

display_game = pygame.display.set_mode((width,height))
pygame.display.set_caption("Number Guesser")
grid = grid(28, 28, width, height)
main()

pygame.quit()
quit()
