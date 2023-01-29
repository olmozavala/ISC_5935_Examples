import matplotlib.pyplot as plt
import numpy as np
from pyecg import ECGRecord
from os.path import join
import os

## Reading available files
# hea_path = "./Data" # If running with F6
hea_path = "./FileFormats/ECG/Data" # If running from console (pycharm)
files = [x for x in os.listdir(hea_path) if x.find(".hea") != -1]
print(files)

## Reading specific file as a ECGRecord object
c_file = files[0]
print(c_file)
record = ECGRecord.from_wfdb(join(hea_path, c_file))
print(f"Record type: {type(record)}")
print(f"Methods: {vars(record)}")

## Reading the signals
fig, axs = plt.subplots(1,1)
signals = record.signals
axs.plot(signals[0])
# axs.plot(signals[0][0:500])
plt.show()

## Reading the annotations
print(record.annotations._labels)

