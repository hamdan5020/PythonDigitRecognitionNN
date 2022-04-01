import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
import numpy as np
from playsound import playsound

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import time
root = tk.Tk()

root.resizable(False, False)

# defines the MLC.
clf = MLPClassifier()

# main window dimensions.
winWidth = 930
winHeight = 555

# grid size.
GridBGDim = 534

# the size of a single pixel on the grid.
rectDim = 32

# stores files for xTr, yTr, xTe, yTe.
textFiles = []

# will be populated with the pixels on the grid.
pixels = []

# pixel offset from edge of grid border.
pixelOfset = 11


# place holder image to show off grid functionality.
img5 = [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,1,1,0,1,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,0,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0
]




# a method that opens the windows file explorer and puts the path of selected file in the text field.
def OpenFile(pathIn, newWindow):
    newWindow.lift()
    filename = filedialog.askopenfilename(initialdir="/", title="Select File")
    newWindow.lift()
    textFiles.append(filename)
    pathIn.insert('1.0', filename)

# a method that opens a new window used to construct the neural network.
def ConsNeuNetWin():
    # creates a new window
    newWindow = tk.Toplevel(root)
    newWindow.geometry("400x220")
    newWindow.resizable(False, False)
    newWindow.grab_set()

    # path for x label
    pathLabelx = tk.Label(newWindow, text="Training X File Path:")
    pathLabelx.grid(column=0, row=0, pady=10)

    # path for x text feild
    pathInx = tk.Text(newWindow, height=1, width=20)
    pathInx.grid(column=1, row=0, pady=10, padx=10)

    # open.. for x Button
    openWinExpBtnx = tk.Button(newWindow, text="Open...", command=lambda: OpenFile(pathInx, newWindow))
    openWinExpBtnx.grid(column=2, row=0, pady=10)

    # path for y label
    pathLabely = tk.Label(newWindow, text="Training Y File Path:")
    pathLabely.grid(column=0, row=1, pady=10)

    # path for y text feild
    pathIny = tk.Text(newWindow, height=1, width=20)
    pathIny.grid(column=1, row=1, pady=10, padx=10)

    # open.. for y Button
    openWinExpBtny = tk.Button(newWindow, text="Open...", command=lambda: OpenFile(pathIny, newWindow))
    openWinExpBtny.grid(column=2, row=1, pady=10)

    # Number of hidden neurons Label
    numNeuLabel = tk.Label(newWindow, text="Number of Hidden Neurons:")
    numNeuLabel.grid(column=0, row=2, pady=10)

    # Number of hidden neurons text box
    numNeuIn = tk.Text(newWindow, height=1, width=20)
    numNeuIn.grid(column=1, row=2, pady=10, padx=10)

    # Learning rate Label
    learnRateLabel = tk.Label(newWindow, text="Learning Rate:")
    learnRateLabel.grid(column=0, row=3, pady=10)

    # Learning rate text box
    learnRateIn = tk.Text(newWindow, height=1, width=20)
    learnRateIn.grid(column=1, row=3, pady=10, padx=10)

    # train model button
    trainmodelBtn = tk.Button(newWindow, text="Train Model", command=lambda: trainModel(pathInx.get("0.0", "end").replace('\n', ''), pathIny.get("0.0", "end").replace('\n', ''), int(numNeuIn.get("0.0", "end").replace('\n', '')), learnRateIn.get("0.0", "end").replace('\n', ''), newWindow))
    trainmodelBtn.grid(column=0, row=4, pady=10)

    # Cancel model button
    trainmodelBtn = tk.Button(newWindow, text="Cancel", fg="red", command=lambda: newWindow.destroy())
    trainmodelBtn.grid(column=1, row=4, pady=10)

# constructs the test model window.
def ConsTestModelWin():
    # creates a new window
    newWindow = tk.Toplevel(root)
    newWindow.geometry("400x150")
    newWindow.resizable(False, False)
    newWindow.grab_set()


    # path for x label
    pathLabelx = tk.Label(newWindow, text="Testing X File Path:")
    pathLabelx.grid(column=0, row=0, pady=10)

    # path for x text feild
    pathInx = tk.Text(newWindow, height = 1, width = 20)
    pathInx.grid(column=1, row=0, pady=10,  padx=10)

    # open.. for x Button
    openWinExpBtnx = tk.Button(newWindow, text="Open...", command= lambda: OpenFile(pathInx, newWindow))
    openWinExpBtnx.grid(column=2, row=0, pady=10)





    # path for y label
    pathLabely = tk.Label(newWindow, text="Testing Y File Path:")
    pathLabely.grid(column=0, row=1, pady=10)

    # path for y text feild
    pathIny = tk.Text(newWindow, height=1, width=20)
    pathIny.grid(column=1, row=1, pady=10, padx=10)

    # open.. for y Button
    openWinExpBtny = tk.Button(newWindow, text="Open...", command=lambda: OpenFile(pathIny, newWindow))
    openWinExpBtny.grid(column=2, row=1, pady=10)

    # test model button
    trainmodelBtn = tk.Button(newWindow, text="Test Model", command=lambda: testModel(pathInx.get("0.0", "end").replace('\n', ''), pathIny.get("0.0", "end").replace('\n', ''), newWindow))
    trainmodelBtn.grid(column=0, row=2, pady=10)

    # Cancel model button
    trainmodelBtn = tk.Button(newWindow, text="Cancel", fg="red", command=lambda: newWindow.destroy())
    trainmodelBtn.grid(column=1, row=2, pady=10)


# main window canvas.
canvas = tk.Canvas(root, height=winHeight, width=winWidth)
canvas.pack()

# settings for the NN button.
constructNNBtn = tk.Button(canvas, text ="Construct Neural Network...")
constructNNBtn.configure(width = 25, command=ConsNeuNetWin)
constructNNBtnWin = canvas.create_window(10, 10, anchor="nw", window=constructNNBtn)

# settings for training time label
timefortrainlabel = tk.Label(canvas, fg="red", text ="Time Taken for Training(Sec): ")
timefortrainlabelWin = canvas.create_window(200, 12, anchor="nw", window=timefortrainlabel)

# the NN graph.
graphBg = canvas.create_rectangle(60, 50, 320, 310, fill="#C4C4C4")

# Settings for the Model Testing Button
TestModelBtn = tk.Button(canvas, text ="Test Model...",state="disabled", command=ConsTestModelWin)
TestModelBtn.configure(width = 10)
TestModelBtnWin = canvas.create_window(10, 325, anchor="nw", window=TestModelBtn)


# config for confusion matrix:
shiftX = 0 # shifts matrix along X-axis
shiftY = 50 # shifts matrix along Y-axis
Tp = 0.0 # true +ive
Fp = 0.0 # false -ive
Fn = 0.0 # false -ive
Tn = 0.0 # true -ive

sens = 0.0 # Sensitivity
spec = 0.0 # Specificity
prec = 0.0 # Precision
NPV = 0.0 # Negative predictive value
acc = 0.0 # Accuracy

# label for True  +ive
TpLabel = tk.Label(canvas, text ="True positive \n", bg="#78e6b4")
TpLabel.configure(width = 10)
TpLabelWin = canvas.create_window(88+ shiftX, 390 + shiftY, anchor="nw", window=TpLabel)

# label for False  +ive
FpLabel = tk.Label(canvas, text ="False positive\n", bg="#e68c8c")
FpLabel.configure(width = 10)
FpLabelWin = canvas.create_window(88+ shiftX, 428 + shiftY, anchor="nw", window=FpLabel)

# label for False  -ive
FnLabel = tk.Label(canvas, text ="False Negative\n", bg="#e68c8c")
FnLabel.configure(width = 10)
FnLabelWin = canvas.create_window(166 + shiftX, 390 + shiftY, anchor="nw", window=FnLabel)

# label for True  -ive
TnLabel = tk.Label(canvas, text ="True Negative\n", bg="#78e6b4")
TnLabel.configure(width = 10)
TnLabelWin = canvas.create_window(166 + shiftX, 428 + shiftY, anchor="nw", window=TnLabel)

# label for Sensitivity
SensLabel = tk.Label(canvas, text ="Sensitivity\n", bg="#808080")
SensLabel.configure(width = 10)
SensLabelWin = canvas.create_window(244 + shiftX, 390 + shiftY, anchor="nw", window=SensLabel)

# label for Specificity
SpecLabel = tk.Label(canvas, text ="Specificity\n", bg="#808080")
SpecLabel.configure(width = 10)
SpecLabelWin = canvas.create_window(244 + shiftX, 428 + shiftY, anchor="nw", window=SpecLabel)

# label for Precision
PrecLabel = tk.Label(canvas, text ="Precision\n", bg="#808080")
PrecLabel.configure(width = 10)
PrecLabelWin = canvas.create_window(88+ shiftX, 466 + shiftY, anchor="nw", window=PrecLabel)

# label for NPV
NPVLabel = tk.Label(canvas, text ="NPV\n", bg="#808080")
NPVLabel.configure(width = 10)
NPVLabelWin = canvas.create_window(166 + shiftX, 466 + shiftY, anchor="nw", window=NPVLabel)

# label for Accuracy
AccLabel = tk.Label(canvas, text ="Accuracy\n", bg="#7896e6")
AccLabel.configure(width = 10)
AccLabelWin = canvas.create_window(244 + shiftX, 466 + shiftY, anchor="nw", window=AccLabel)


# Neural network Setup:

# count the number of lines in a file
def countNumOfLines(path):
    num_lines = sum(1 for line in open(path))
    return num_lines


# converts .the txt file to a 2D array to be accepted by the ML algorithm.
def convTxtTo2DArr(path):
    arrayofNums = [[0 for x in range(16*16)] for y in range(countNumOfLines(path))]
    textFile = open(path)
    for i in range(len(arrayofNums)):
        tempstr = textFile.readline()
        tempstr = tempstr.replace(',', '')
        for j in range(len(arrayofNums[i])):
            if (tempstr[j] == '1' or tempstr[j] == '0'):
                arrayofNums[i][j] = int(tempstr[j])
    textFile.close()
    return arrayofNums


# converts the .txt file to a Y array to be accepted by the ML algorithm.
def convYTxtToArrs(path):
    textFile = open(path)
    arrayofnums = []
    for i in range(countNumOfLines(path)):
        arrayofnums.append(textFile.readline().replace('\n', ''))
    return arrayofnums

# a method that gets called by the button consNewNNBtn to start training.
def trainModel(pathx, pathy, numOfNeu, learnRate, newwin):
    xTrain = convTxtTo2DArr(pathx)
    yTrain = convYTxtToArrs(pathy)
    newwin.destroy()
    global clf
    start = time.time()
    clf = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(numOfNeu, ))
    clf.fit(xTrain, yTrain)
    end = time.time()
    global timefortrainlabel
    timefortrainlabel.configure(text="Time Taken for Training(Sec): " + str(end - start)[0:4])
    TestModelBtn.configure(state="normal")
    DrawBtn.configure(state="normal")

# a method that tests a loaded model, fired by "Test Model" button.
def testModel(pathx, pathy, window):
    global Fp, Fn, Tp, Tn, sens, spec, prec, NPV , acc
    xTest = convTxtTo2DArr(pathx)
    yTest = convYTxtToArrs(pathy)
    window.destroy()
    predict = clf.predict(xTest)
    CM = confusion_matrix(yTest, predict)

    Tn = CM[0][0]
    Fn = CM[1][0]
    Tp = CM[1][1]
    Fp = CM[0][1]

    sens = Tp/(Tp+Fn)
    spec = Tn/(Tn+Fp)
    prec = Tp/(Tp+Fp)
    NPV = Tn/(Tn+Fn)
    acc = findAcc(CM)
    updateLabels()

# method that updates the labels of the confusion matrix to have values after testing
def updateLabels():

    TpLabel.configure(text=TpLabel.cget("text") + str(Tp))
    FpLabel.configure(text=FpLabel.cget("text") + str(Fp))

    TnLabel.configure(text=TnLabel.cget("text") + str(Tn))
    FnLabel.configure(text=FnLabel.cget("text") + str(Fn))

    SensLabel.configure(text=SensLabel.cget("text") + str(sens)[0:5])
    SpecLabel.configure(text=SpecLabel.cget("text") + str(spec)[0:5])
    PrecLabel.configure(text=PrecLabel.cget("text") + str(prec)[0:5])
    NPVLabel.configure(text=NPVLabel.cget("text") + str(NPV)[0:5])

    AccLabel.configure(text=AccLabel.cget("text") + str(acc))

# a method that finds the accuracy of the model after testing
def findAcc(confusion_matrix):
    diag = confusion_matrix.trace()
    elem = confusion_matrix.sum()
    return diag/elem


pressed = False

# toggles pen down or pen up by pressing the space bar.
def ToggleDraw(event):
    if event.keysym == "space":
        global pressed
        pressed = not pressed

root.bind('<KeyPress>', ToggleDraw)


# class that makes pixels (Rectangles) in the grid clickable and change color
class Pixel():

    pixelColor = "black"
    def __init__(self, can, start_x, start_y, size):
        self.can=can
        self.id = self.can.create_rectangle((start_x, start_y,
                  start_x+size, start_y+size), fill="black")

        self.can.tag_bind(self.id, "<Enter>", self.set_color)
        self.color_change = True

    # a method that resets the color of a pixel to black, fired by "Clear Grid" Button
    def ResetColor(self):
        global pixelColor
        self.pixelColor = "black"

        self.can.itemconfigure(self.id, fill="black")

    # a method that is called to specifically be used to change the color of a pixel (USED FOR DEBUGGING ONLY NOT FOR
    # REGULAR PROGRAM USE)
    def SetColor(self, color):
        global pixelColor
        self.pixelColor = color

        self.can.itemconfigure(self.id, fill=color)

    # a method that sets the color of pixel, fired by the Enter event.
    def set_color(self, event=None):
        if pressed:
            global pixelColor
            self.pixelColor="white"
            self.can.itemconfigure(self.id, fill="white")

    # a method that returns the color of a particular pixel.
    def getPixelColor(self):
        return self.pixelColor

# Builds the grid that numbers will be displayed on.
GridBGRec = canvas.create_rectangle(winWidth - GridBGDim - 10, 10, winWidth - GridBGDim + GridBGDim - 10, GridBGDim + 10, fill="#C4C4C4")
for i in range(0, 16):
    for j in range(0, 16):
        pixels.append(Pixel(canvas, canvas.coords(GridBGRec)[0] + pixelOfset + 32*i,canvas.coords(GridBGRec)[1] + pixelOfset + 32*j ,32 ))

# a method that converts a number from grid representation to array representation for NN to predict.
def PixelsToArr():
    imgArr = [0 for i in range(256)]
    for i in range(len(pixels)):
        if pixels[i].getPixelColor() == "white":
            imgArr[i] = 1
    return imgArr

# method to predict and play sound.
def predictImgOnGrid():
    digitSoundArr = ["num0.mp3", "num1.mp3", "num2.mp3", "num3.mp3", "num4.mp3", "num5.mp3", "num6.mp3", "num7.mp3",
                     "num8.mp3", "num9.mp3"]

    imgArr = np.array( PixelsToArr())
    predict = clf.predict(imgArr.reshape(1, -1))
    num = int(predict[0])
    playsound(digitSoundArr[num])


# Settings for the Draw Button
DrawBtn = tk.Button(canvas, text ="Predict Digit On Grid", command=predictImgOnGrid, state="disable")
DrawBtn.configure(width = 25)
DrawBtnWin = canvas.create_window(100, 325, anchor="nw", window=DrawBtn)

# a method that clears the grid i.e. sets the color of every pixel to black, fired by the "Clear Grid" button.
def clearGrid():
    for i in range(len(pixels)):
        pixels[i].ResetColor()


# Settings for the Clear Grid Button
ClearBtn = tk.Button(canvas, text ="Clear Grid", fg="red", command=clearGrid)
ClearBtn.configure(width = 10)
ClearBtnWin = canvas.create_window(300, 325, anchor="nw", window=ClearBtn)


# Important tip label settings.
tiplabel = tk.Label(canvas, fg="red", text ="Important Tip: To draw a number: press (space bar) to put the pen\n down  "
                                            "then, move mouse on grid. when done, press (space bar) again\n to lift"
                                            "  pen. Then press \"Predict Digit On Grid\" button. The program\n will"
                                            " speak out the predicted digit. "
                                            "\nTo reset Grid press the \"Clear Grid\" Button")

tiplabelWin = canvas.create_window(10, 352, anchor="nw", window=tiplabel)

# method used to transfer an array of 1's and 0's to a grid representation.
def displayNumOnGrid(img):
    for i in range(0, 256):
        if img[i] == 1:
            pixels[i].SetColor("white")

displayNumOnGrid(img5)

root.mainloop()



