import numpy as np
import csv
import random


csv_file = 'DSSLFS(Dataset1)(64dim).csv'
txt_file = 'DSSLFS(Dataset1)(64dim).txt'
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [my_output_file.write(" ".join(row) + '\n') for row in csv.reader(my_input_file)]
    my_output_file.close()

csv_file = 'DGSLGS(Dataset1)(64dim).csv'
txt_file = 'DGSLGS(Dataset1)(64dim).txt'
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [my_output_file.write(" ".join(row) + '\n') for row in csv.reader(my_input_file)]
    my_output_file.close()

csv_file = 'DCSLCS(Dataset1)(64dim).csv'
txt_file = 'DCSLCS(Dataset1)(64dim).txt'
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [my_output_file.write(" ".join(row) + '\n') for row in csv.reader(my_input_file)]
    my_output_file.close()


with open("DSSLFS(Dataset1)(64dim).txt") as dss:
    with open("DGSLGS(Dataset1)(64dim).txt") as dgs:
        with open("DCSLCS(Dataset1)(64dim).txt") as dcs:
            with open("Feature(Dataset1)(original).txt", "w") as pf:
                dsslines = dss.readlines()
                dgslines = dgs.readlines()
                dcslines = dcs.readlines()
                for k in range(len(dsslines)):
                    line = dsslines[k].strip() + ' ' + dgslines[k].strip() + ' ' +dcslines[k]
                    pf.write(line)

SampleFeature = np.loadtxt('Feature(Dataset1)(original).txt',dtype=np.float64)

SampleLabel = []
counter = 0
while counter < len(SampleFeature) / 2:
    SampleLabel.append(1)
    counter = counter + 1
counter1 = 0
while counter1 < len(SampleFeature) / 2:
    SampleLabel.append(0)
    counter1 = counter1 + 1

# shuffle the dataset
counter = 0
R = []
while counter < len(SampleFeature):
    R.append(counter)
    counter = counter + 1
random.shuffle(R)
RSampleFeature = []
RSampleLabel = []
counter = 0
while counter < len(SampleFeature):
    RSampleFeature.append(SampleFeature[R[counter]])
    RSampleLabel.append(SampleLabel[R[counter]])
    counter = counter + 1
RSampleFeature=np.array(RSampleFeature)
RSampleLabel=np.array(RSampleLabel)

np.savetxt('Feature(Dataset1).txt',RSampleFeature,fmt='%f',delimiter=' ')
np.savetxt('Label(Dataset1).txt',RSampleLabel,fmt='%d',delimiter=' ')

