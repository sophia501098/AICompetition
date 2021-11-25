import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
class myDataSet(data.Dataset):
    def __init__(self,phase):
        super().__init__()
        if phase=="train":
            self.datapath="../image/train/"
        else:
            self.datapath="../image/test/"
        self.datalistpath="../train.labels.csv"
        self.datalist=pd.read_csv(self.datalistpath)

    def __len__(self):
        return len(self.Data)

    #得到数据内容和标签
    def __getitem__(self, index):
        curline=self.datalist.iloc[index,0]
        str_list = curline.split("\t")
        filename=str_list[0]
        label=str_list[1]
        data = cv2.imread(self.datapath+filename)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = torch.from_numpy(data)
        return data, label

