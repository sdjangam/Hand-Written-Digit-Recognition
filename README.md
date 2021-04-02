# ML Mini-Project---Hand-Written-Digit-Recognition

# HAND WRITTEN DIGIT RECOGNITION USING TENSORFLOW AND PYTHON

# INTRODUCTION

WHAT IS TENSORFLOW ?

IT IS A MACHINE LEARNING LIBRARY INTRODUCED BY GOOGLE.


WHAT IS MNIST DATA SET ?
IT IS A SUBSET OF NIST DATA SET WHICH CONTAINS(A-Z),(A-Z),(0-9) ALL BLACK AND WHITE IMAGES.

IT IS DATASET OF HANDWRITTEN DIGITS(0-9).

IT CONSIST OF 55000 TRAINING DATA AND 10000 TEST DATA.


DATA PREPROCESSING OF MNIST

INITIALLY THE IMAGES WERE NORMALIZED TO 20X20 PIXELS AND CENTERED , ALSO THEIR ASPECT RATIO WAS
MAINTAINED.

AFTER THIS STEP THEY WERE RESIZED TO 28X28 PIXELS IN ORDER TO GET BETTER ACCURACY AND DISTINCTION
BETWEEN THE BLACK AND WHITE PIXELS IN THE IMAGE.


# INPUT DATA

THE INPUT TO THE MODEL IS THE PIXEL DATA SHOWN AND THE ARRAY OF VALUES SHOWN ABOVE
THESE ARRAY VALUES ARE THE NUMBERS EACH IMAGE REPRESENT.

THE NUMBERS REPRESENTED BY THE ARRAY ARE STORED IN THE FORM OF 'ONE HOT ENCODING' I.E.
THE POSITION IN THE ARRAY WHERE ONE IS PRESENT REPRESENTS THE NUMBER PRRESENT IN THE IMEGES.

# APPROACH

A PERCEPTRON IS A NODE WHICH TAKES INPUT PROCESSES IT AND GIVES SINGLE OUTPUT.

SINGLE LAYER OF PERCEPTRON IS A NEURAL NETWORK.

MULTIPLE LAYERS OF PERCEPTRON(>=2) IS CALLED A DEEP NEURAL NETWORK.

GIVES BETTER ACCURACY COMPARED OTHER ALGORITHMS LIKE LINEAR REGRESSION

SOME OF THE ML ALGORITHMS LIKE DECISION TREES CANNOT GIVE HIGH ACCURACY ON

MNIST DATA DUE TO IT'S LIMITATION OF PROCESSING HIGH DIMENSIONAL DATA.

INITIALLY I USED 3-HIDDEN LAYER EACH CONTAINING 500 NODES AND
RAN FOR 10 EPOCHS

THEN I USED THE SAME MODEL AND RAN FOR 15 EPOCHS

THEN I USED 4-HIDDEN LAYERS EACH CONTAINING 500,1500,1500,500

NODES RESPECTIVELY FOR 10 EPOCHS

THEN I USED THE SAME MODEL AND RAN FOR 20 EPOCHS

MY FINAL MODEL IS THE SAME 4-HIDDEN LAYER MODEL WHICH I RAN ON 

15 EPOCHS.


# TERMS

RELU

RECTIFIED LINEAR UNIT GIVES SMOOTH APPROXIMATION  f(x) = ln(1+e^x)

CROSS ENTROPY LOSS

FUNCTION USED TO MEASURE ERROR AT SOFTMAX LAYER

SOFTMAX

ACTIVATION FUNCTION/LAYER WHICH INTERPRETS OUTPUT AS PROBABILITIES

ADAM OPTIMIZER


OPTIMIZER SIMILAR TO GRADIENT DESCENT OPTIMIZER BUT GIVES MUCH EFFICIENT RESULTS FOR AN
EPOCH(FEED FORWARD AND BACKPROPAGATE )

THE ACCURACY FOR MY FINAL
MODEL AS PER THE PREVIOUS
PICTURE IS 96.28%.

ALSO THE OUTPUT OBTAINED FOR
TEST DATA IS IN THE FORM OF 'ONE
HOT ENCODING'.

# FUTURE SCOPE

EXTEND THE MODEL TO WORK ON NIST DATASET

INCREASE THE ACCURACY FURTHER BY IMPLEMENTING MORE NUMBER OF HIDDEN LAYERS
AND/OR EPOCHS

DETECT CUSTOM HAND WRITTEN DIGITS

USE CNN WITH LESS LAYERS TO GET BETTER ACCURACY

# DISCUSSION

RANDOM FOREST GIVES AN ACCURACY OF 0.8 APPROX FOR THE MNIST DATA SET

KNN ALGORITHM GIVES AN ACCURACY OF 0.94 APPROX FOR THE MNIST DATA SET

DNN GIVES AN ACCURACY OF 0.962 APPROX FOR THE MNIST DATA SET

CNN CAN BE USED TO ACHIEVE AN ACCURACY OF 0.992 APPROX FOR THE MNIST DATA SET

THUS NEURAL NETWORKS CAN GIVE BETTER AND MORE ACCURATE RESULTS FOR THIS PARTICULAR

PROBLEM COMPARED TO OTHER MACHINE LEARNING ALGORITHMS

