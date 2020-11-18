import cv2
import numpy as np
import pandas as pd
import pytesseract
from matplotlib import pyplot as plt
from difflib import SequenceMatcher


def shadow_remove(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    shadowremov = cv2.merge(result_norm_planes)
    return shadowremov

def simpleth(img,method):
    if method == 1:
        ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    elif method == 2:
        ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    elif method == 3:
        ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    elif method == 4:
        ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    elif method == 5:
        ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    return th

def adaptiveth(img,method):
    if method == 1:
        ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    elif method == 2:
        th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    elif method == 3:
        th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return th

def otsu(img,method):
    if method == 1:
        ret,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 2:
        ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == 3:
        ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
    elif method == 4:
        ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    elif method == 5:
        ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)
    return th

def detect_char(img):
    imgt=img
    hImg,wImg = imgt.shape
    boxes = pytesseract.image_to_boxes(imgt)
    for b in boxes.splitlines():
        b = b.split(' ')
        x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
        cv2.rectangle(imgt,(x,hImg-y),(w,hImg-h),(0,0,255),1)
        cv2.putText(imgt,b[0],(x,hImg-y+8),cv2.FONT_HERSHEY_COMPLEX,0.4,(50,50,255),1)
    return imgt

def detect_word(img):
    imgt=img
    #hImg, wImg = img.shape
    boxes = pytesseract.image_to_data(imgt)
    for x, b in enumerate(boxes.splitlines()):
        if x != 0:
            b = b.split()
            if len(b) == 12:
                x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                cv2.rectangle(imgt, (x, y), (w + x, h + y), (0, 0, 255), 1)
                cv2.putText(imgt, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (50, 50, 255), 1)
    return imgt

def evalAcc (img,corr):
    predStr = pytesseract.image_to_string(img)
    sim = SequenceMatcher(None, corr, predStr).ratio()
    return sim

def main():
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'
    img = cv2.imread('sample01.png')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corrStr1 = "Parking: You may park anywhere on the campus where there are no signs prohibiting par-\nking. Keep in mind the carpool hours and park accordingly so you do not get blocked in the\nafternoon\n\nUnder School Age Children:While we love the younger children, it can be disruptive and\ninappropriate to have them on campus during school hours. There may be special times\nthat they may be invited or can accompany a parent volunteer, but otherwise we ask that\nyou adhere to our    policy for the benefit of the students and staff."
    corrStr2 = "Sonnet for Lena\n\nO dear Lena, your beauty is so vast\nIt is hard sometimes to describe it fast.\nI thought the entire world I would impress\nIf only your protrait I could compress.\nAlas! First when I tried to use VQ\nI found that your cheeks belong to only you.\nYour silky hair contains a thousand lines\nHard to match with sums of discrete cosines.\nAnd for your lips, sensual and tactual\nThirteen Crays found not the proper fractal.\nAnd while these setbacks are all quite severe\nI might have fixed them with hacks here and there\nBut when filters took sparkle from your eyes\nI said, 'Damn all this. I'll just digitize.'\n\nThomas Cothurst"
    corr = corrStr1

    #Simple Thresholding

    #Method numbers: 1-Binary, 2-BinaryInv, 3-Trunc, 4-ToZero, 5-ToZeroInv
    sth1 = simpleth(img_gray,1)
    sth2 = simpleth(img_gray,2)
    sth3 = simpleth(img_gray,3)
    sth4 = simpleth(img_gray,4)
    sth5 = simpleth(img_gray,5)

    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [img_gray, sth1, sth2, sth3, sth4, sth5]

    simpleTh = plt.figure('Simple Threshold')
    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    #Otsu Binarization for Simple Thresholding

    #Method numbers: 1-Binary, 2-BinaryInv, 3-Trunc, 4-ToZero, 5-ToZeroInv
    osth1 = otsu(img_gray,1)
    osth2 = otsu(img_gray,2)
    osth3 = otsu(img_gray,3)
    osth4 = otsu(img_gray,4)
    osth5 = otsu(img_gray,5)

    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [img_gray, osth1, osth2, osth3, osth4, osth5]

    simpleTh_Otsu = plt.figure('Simple Threshold (Otsu)')
    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])


    #Adaptive Thresholding

    #Method numbers: 1-NoAdaptive, 2-MeanAdaptive, 3-GaussianAdaptive
    ath1 = adaptiveth(img_gray,1)
    ath2 = adaptiveth(img_gray,2)
    ath3 = adaptiveth(img_gray,3)

    titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, ath1, ath2, ath3]

    adaptiveTh = plt.figure('Adaptive Threshold')
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    #Adaptive Thresholding with blur

    img_blur = cv2.medianBlur(img_gray, 5)

    #Method numbers: 1-NoAdaptive, 2-MeanAdaptive, 3-GaussianAdaptive
    bath1 = adaptiveth(img_blur,1)
    bath2 = adaptiveth(img_blur,2)
    bath3 = adaptiveth(img_blur,3)

    titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, bath1, bath2, bath3]

    adaptiveThBlur = plt.figure('Adaptive Threshold with Blur')
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])


    #Best Result
    shad = shadow_remove(img_gray)



    # #OCR accuracy evaluation

    #Simple Threshold evaluation
    titles = ['Similarity Table','Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    sim1 = ['Without Otsu',img_gray, sth1, sth2, sth3, sth4, sth5]
    sim2 = ['With Otsu',img_gray, osth1, osth2, osth3, osth4, osth5]

    for i in range(6):
        sim1[i+1] = round(evalAcc(sim1[i+1],corr),3)

    for i in range(6):
        sim2[i+1] = round(evalAcc(sim2[i+1],corr),3)

    simtable = [titles,sim1,sim2]
    simtable_t = np.array(simtable).T

    print(simtable_t)

    #Adaptive Thresholding Evaluation
    titles = ['Similarity Table', 'Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    sim1 = ['Without blur', img, ath1, ath2, ath3]
    sim2 = ['With blur', img, bath1, bath2, bath3]

    for i in range(4):
        sim1[i + 1] = round(evalAcc(sim1[i + 1], corr),3)

    for i in range(4):
        sim2[i + 1] = round(evalAcc(sim2[i + 1], corr),3)

    simtable = [titles, sim1, sim2]
    simtable_t = np.array(simtable).T

    print(simtable_t)

    print(round(evalAcc(shad, corr),3))


    #Detecting Characters
    dsth1 = detect_char(sth1)
    dath2 = detect_char(ath2)
    dshad = detect_char(shad)

    titles = ['STH1', 'ATH2',
                'Shad']
    images = [dsth1, dath2, dshad]

    characters = plt.figure('Detecting Characters')
    for i in range(3):
        plt.subplot(3,1,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    # Detecting words
    dsth1 = detect_word(sth1)
    dath2 = detect_word(ath2)
    dshad = detect_word(shad)

    titles = ['STH1', 'ATH2',
              'Shad']
    images = [dsth1, dath2, dshad]

    characters = plt.figure('Detecting Words')
    for i in range(3):
        plt.subplot(3, 1, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])


        plt.show()
        cv2.waitKey(0)

if __name__ == "__main__":
    main()