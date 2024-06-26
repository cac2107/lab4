import os
import random
from PIL import Image, ImageStat
from tqdm import tqdm
import math

def extract_txt_details(img: str):
    txt = img.replace(".png", ".txt")

    with open(txt, 'r') as f: lines = f.readlines()
    gender = lines[0].split(":")[1].strip()
    f_class = lines[1].split(":")[1].strip()

    return gender, f_class

def compare_prints(imageA, imageB):
    imA = Image.open(imageA)
    imB = Image.open(imageB)
    stat1 = ImageStat.Stat(imA)
    stat2 = ImageStat.Stat(imB)
    diff = math.fabs(stat1.sum2[0] - stat2.sum2[0])/(stat1.sum2[0])
    return(1-diff)

def for_hybrid(file1, file2, expv=0.34):
    result = 0
    try: result = compare_prints(file1, file2)
    except Exception as e: print(f"Error in pillow sum: {e}")

    return result >= expv

def check_result(file1, file2):
    """
    (pos, negs, c_pos, f_neg, f_pos, c_neg)
    """
    pos = 0
    negs = 0
    correctpos = 0
    falseneg = 0
    falsepos = 0
    correctneg = 0
    posbool = False

    try: result = compare_prints(os.path.join("test", file1), os.path.join("test", file2))
    except Exception as e: print(f"Exception: {e}")

    if result >= 0.34:
        pos += 1
        posbool = True
    else: negs += 1

    if file1[1:] == file2[1:]:
        if posbool: correctpos += 1
        else: falseneg += 1
    else:
        if posbool: falsepos += 1
        else: correctneg += 1

    return (pos, negs, correctpos, falseneg, falsepos, correctneg)

def main():
    correctpos = 0
    falsepos = 0
    falseneg = 0
    correctneg = 0
    negs = 0
    pos = 0

    files = os.listdir("test")
    files2 = []
    for f in files:
        if f.endswith(".png"): files2.append(f)
    files = files2

    sample = random.sample(files, 200)

    for i in tqdm(range(len(sample)), desc="Processing..."):
        file1 = sample[i]
        gender1, class1 = extract_txt_details(os.path.join("test", file1))

        for j in range(len(files)):
            file2 = files[j]

            gender2, class2 = extract_txt_details(os.path.join("test", file2))
            if not (gender1 == gender2 and class1 == class2):
                negs += 1
                correctneg += 1
                continue
            
            ps, ns, cp, fn, fp, cn = check_result(file1, file2)
            pos += ps
            negs += ns
            correctpos += cp
            falseneg += fn
            falsepos += fp
            correctneg += cn

    print(f"Total Positives: {pos}\nTotal Negs: {negs}\nCorrect Pos: {correctpos}\nCorrect Neg: {correctneg}\nFalse Pos: {falsepos}\nFalse Neg: {falseneg}")

if __name__ == "__main__":
    main()