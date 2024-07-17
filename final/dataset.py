import pandas as pd
import shutil, os

pwd = "/home/parth/dataset/"

df = pd.read_csv("/home/parth/dataset/train.csv")

for i in range(len(df)):
	dest_folder = pwd + df["target"][i]
	source_file = pwd + "train/" +  df["Image"][i]
	shutil.move(source_file, dest_folder)
	print(i)




