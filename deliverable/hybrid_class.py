import os
import random
import orbt
import perrrms
import pillowsum

from tqdm import tqdm

class hybridc:
    rmsv = 0
    sumv = 0
    orbv = 0

    correctpos = 0
    falsepos = 0
    falseneg = 0
    correctneg = 0
    negs = 0
    pos = 0
    frr = 0
    far = 0

    def extract_txt_details(self, img: str):
        txt = img.replace(".png", ".txt")

        with open(txt, 'r') as f: lines = f.readlines()
        gender = lines[0].split(":")[1].strip()
        f_class = lines[1].split(":")[1].strip()

        return gender, f_class
    
    def calc_frr(self, expected_pos): self.frr = self.falseneg / expected_pos
    def calc_far(self, expected_neg): self.far = self.falsepos / expected_neg

    def train(self):
        files = os.listdir("train")
        files2 = []
        for f in files:
            if f.endswith(".png"): files2.append(f)
        files = files2

        rmspos = []
        sumpos = []
        orbpos = []

        sample = random.sample(files, 400)

        for i in tqdm(range(len(sample)), desc="Training... "):
            file1 = sample[i]
            f1path = os.path.join("train", file1)
            gender1, class1 = self.extract_txt_details(f1path)

            for j in range(len(files)):
                file2 = files[j]
                f2path = os.path.join("train", file2)
                gender2, class2 = self.extract_txt_details(f2path)
                if not (gender1 == gender2 and class1 == class2): continue

                if file1[1:] == file2[1:]:
                    rmspos.append(perrrms.compare_prints(f1path, f2path))
                    sumpos.append(pillowsum.compare_prints(f1path, f2path))
                    orbpos.append(orbt.compare_prints(f1path, f2path))

        rmspos.sort()
        sumpos.sort()
        orbpos.sort()

        self.rmsv = rmspos[len(rmspos)//8]
        self.sumv = sumpos[len(sumpos)//6]
        self.orbv = orbpos[len(orbpos)//4]
        print(f"rmsv: {self.rmsv:.5f} | sumv: {self.sumv:.5f} | orbv: {self.orbv:.5f}")
    
    def hybrid_check(self, file1, file2):
        f1path = os.path.join("test", file1)
        f2path = os.path.join("test", file2)
        orb_result = orbt.for_hybrid(f1path, f2path, expv=self.orbv)
        sum_result = pillowsum.for_hybrid(f1path, f2path, expv=self.sumv)
        rms_result = perrrms.for_hybrid(f1path, f2path, expv=self.rmsv)

        posbool = False

        if sum([orb_result, sum_result, rms_result]) >= 2:
            posbool = True
            self.pos += 1
        else: self.negs += 1

        if file1[1:] == file2[1:]:
            if posbool: self.correctpos += 1
            else: self.falseneg += 1
        else:
            if posbool: self.falsepos += 1
            else: self.correctneg += 1

    def test(self):
        files = os.listdir("test")
        files2 = []
        for f in files:
            if f.endswith(".png"): files2.append(f)
        files = files2

        for i in tqdm(range(len(files)), desc="Testing... "):
            file1 = files[i]
            f1path = os.path.join("test", file1)
            gender1, class1 = self.extract_txt_details(f1path)
            self.hybrid_check(file1, file1)

            for j in range(i+1, len(files)):
                file2 = files[j]
                f2path = os.path.join("test", file2)

                gender2, class2 = self.extract_txt_details(f2path)
                if not (gender1 == gender2 and class1 == class2):
                    self.negs += 1
                    self.correctneg += 1
                    continue

                self.hybrid_check(file1, file2)

    def final_calcs(self):
        expected_pos = self.correctpos + self.falseneg
        expected_neg = self.correctneg + self.falsepos
        self.calc_far(expected_neg)
        self.calc_frr(expected_pos)

    def main(self):
        self.train()
        self.test()
        self.final_calcs()
        print(f"Total Positives: {self.pos}\nTotal Negs: {self.negs}\nCorrect Pos: {self.correctpos}\nCorrect Neg: {self.correctneg}\nFalse Pos: {self.falsepos}\nFalse Neg: {self.falseneg}")
        print(f"FRR: {self.frr:.5f}\nFAR: {self.far:.5f}")

def main():
    hc = hybridc()
    hc.main()

if __name__ == "__main__":
    main()