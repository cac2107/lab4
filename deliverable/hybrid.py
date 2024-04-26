import os
import random
import orbt
import perrrms
import pillowsum

from tqdm import tqdm

def extract_txt_details(img: str):
    txt = img.replace(".png", ".txt")

    with open(txt, 'r') as f: lines = f.readlines()
    gender = lines[0].split(":")[1].strip()
    f_class = lines[1].split(":")[1].strip()

    return gender, f_class

def train():
    files = os.listdir("train")
    files2 = []
    for f in files:
        if f.endswith(".png"): files2.append(f)
    files = files2

    rmspos = []
    sumpos = []
    orbpos = []

    sample = random.sample(files, 200)

    for i in tqdm(range(len(sample)), desc="Training... "):
        file1 = sample[i]
        f1path = os.path.join("train", file1)
        gender1, class1 = extract_txt_details(f1path)

        for j in range(len(files)):
            file2 = files[j]
            f2path = os.path.join("train", file2)
            gender2, class2 = extract_txt_details(f2path)
            if not (gender1 == gender2 and class1 == class2): continue

            if file1[1:] == file2[1:]:
                rmspos.append(perrrms.compare_prints(f1path, f2path))
                sumpos.append(pillowsum.compare_prints(f1path, f2path))
                orbpos.append(orbt.compare_prints(f1path, f2path))

    rmspos.sort()
    sumpos.sort()
    orbpos.sort()

    rms_q1 = rmspos[len(rmspos)//4]
    sum_q1 = sumpos[len(sumpos)//4]
    orb_q1 = orbpos[len(orbpos)//4]
    print(f"First quartile rms: {rms_q1}")
    print(f"First quartile sum: {sum_q1}")
    print(f"First quartile orb: {orb_q1}")

    return rms_q1, sum_q1, orb_q1

def hybrid_check(file1, file2):
    f1path = os.path.join("test", file1)
    f2path = os.path.join("test", file2)
    orb_result = orbt.for_hybrid(f1path, f2path)
    sum_result = pillowsum.for_hybrid(f1path, f2path)
    rms_result = perrrms.for_hybrid(f1path, f2path)

    posbool = False
    pos = 0
    negs = 0
    correctpos = 0
    falseneg = 0
    falsepos = 0
    correctneg = 0

    if sum([orb_result, sum_result, rms_result]) >= 2:
        posbool = True
        pos += 1
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

    for i in tqdm(range(len(sample)), desc="Testing... "):
        file1 = sample[i]
        f1path = os.path.join("test", file1)
        gender1, class1 = extract_txt_details(f1path)

        for j in range(len(files)):
            file2 = files[j]
            f2path = os.path.join("test", file2)

            gender2, class2 = extract_txt_details(f2path)
            if not (gender1 == gender2 and class1 == class2):
                negs += 1
                correctneg += 1
                continue
            
            ps, ns, cp, fn, fp, cn = hybrid_check(file1, file2)
            pos += ps
            negs += ns
            correctpos += cp
            falseneg += fn
            falsepos += fp
            correctneg += cn

    print(f"Total Positives: {pos}\nTotal Negs: {negs}\nCorrect Pos: {correctpos}\nCorrect Neg: {correctneg}\nFalse Pos: {falsepos}\nFalse Neg: {falseneg}")

if __name__ == "__main__":
    main()