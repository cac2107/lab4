# Alex Rosse
# CSEC 472 - Lab 3
# Group 2
# Method 1

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image
from tqdm import tqdm

def detect(detector, image):
    finger = cv2.imread(image)
    finger = cv2.cvtColor(finger, cv2.COLOR_BGR2RGB)
    key_points, des = detector.detectAndCompute(finger, None)
    return finger, key_points, des

def compare_prints(image1, image2):
    detector = cv2.ORB_create() 
    fng1, kp1, des1 = detect(detector, image1)
    fng2, kp2, des2 = detect(detector, image2)
  
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
   
    count = 0 
    for i in matches:
        count += 1

    return count/500

def extract_txt_details(img: str):
    txt = img.replace(".png", ".txt")

    with open(txt, 'r') as f: lines = f.readlines()
    gender = lines[0].split(":")[1].strip()
    f_class = lines[1].split(":")[1].strip()

    return gender, f_class

def test():
    print(compare_prints("test/f1501_06.png", "test/f1501_06.png"))

def train2():
    files = os.listdir("train")
    p_results = []
    n_results = []

    for i in tqdm(range(len(files)), desc="Training..."):
        file1 = files[i]
        if file1[0] == "s" or file1.endswith(".txt"): continue
        gender1, class1 = extract_txt_details(os.path.join("train", file1))

        for j in range(i+1, len(files)):
            file2 = files[j]
            if file2.endswith(".txt"): continue
            result = 0

            gender2, class2 = extract_txt_details(os.path.join("train", file2))
            if not (gender1 == gender2 and class1 == class2): continue

            try: result = compare_prints(os.path.join("train", file1), os.path.join("train", file2))
            except Exception as e: print(f"Exception in Train: {e}")

            if(file1[1:] == file2[1:]):
                p_results.append(result)
            else: n_results.append(result)

        if i%5 == 0:
            p_results.sort()
            n_results.sort()
            with open(f"train_results/nresults{i}.txt", 'w') as f: f.write(str(n_results))
            with open(f"train_results/presults{i}.txt", 'w') as f: f.write(str(p_results))

    p_results.sort()
    n_results.sort()
    print(f"Len p_results: {len(p_results)} | Len n_results: {len(n_results)}")
    print(f"First quartile p: {p_results[len(p_results)//4]} | n: {n_results[len(n_results)//4]}")
    print(f"Median p: {p_results[len(p_results)//2]} | n: {p_results[len(p_results)//2]}")

def train():
    files = os.listdir("train")
    results = []

    for i in tqdm(range(len(files)), desc="Training..."):
        file1 = files[i]
        if file1[0] == "s" or file1.endswith(".txt"): continue

        file2 = "s" + file1[1:]

        result = 0
        try: result = compare_prints(os.path.join("train", file1), os.path.join("train", file2))
        except Exception as e: print(f"Exception in Train: {e}")

        results.append(result)

    results.sort()
    print(f"Len results: {len(results)}")
    print(f"First quartile: {results[len(results)//4]}")
    print(f"Median: {results[len(results)//2]}")


def main():
    correctpos = 0
    falsepos = 0
    falseneg = 0
    correctneg = 0
    negs = 0
    pos = 0

    files = os.listdir("test")

    for i in range(len(files)):
        file1 = files[i]
        if file1.endswith(".txt"): continue
        gender1, class1 = extract_txt_details(file1)

        for j in range(i+1, len(files)):
            file2 = files[j]
            if file2.endswith(".txt"): continue

            posbool = False
            txtmatch = False
            gender2, class2 = extract_txt_details(file1)
            if gender1 == gender2 and class1 == class2: txtmatch = True

            try: result = compare_prints(os.path.join("test", file1), os.path.join("test", file2))
            except Exception as e: print(f"Exception: {e}")

            if result >= 0.34 and txtmatch:
                pos += 1
                posbool = True
            else: negs += 1

            if file1[1:] == file2[1:] and txtmatch:
                if posbool: correctpos += 1
                else: falseneg += 1
                print(f"POSITIVE: {result}")
            else:
                if posbool: falsepos += 1
                else: correctneg += 1
    print(f"Total Positives: {pos}\nTotal Negs: {negs}\nCorrect Pos: {correctpos}\n Correct Neg: {correctneg}\nFalse Pos: {falsepos}\nFalse Neg: {falseneg}")

train2()