Project Title: Text Image Segmentation for Optimal Optical character recognition

Motivation: Contribution to the advancement of OCR technology

Features:
-Simple thresholding
-Simple thresholding with Otsu algorithm
-Adaptive thresholding
-Shadow removal method
-Usage of Tesseract OCR for accuracy calculation and visualization

Installation:
-opencv
-numpy
-pytesseract

Program Execution:
Once the above are installed, simply run the program.
The images at each timestep will be shown in figures annd the accuracy will be printed in the terminal.

To change between sample01 and sample02:
-Change hyperparameters in adaptiveth() function and shadow_remove() function according to the numbers in the cimments.
-Change the input image under main
-change the variable corr to corrStr1 for sample01 and corrStr2 for sample02