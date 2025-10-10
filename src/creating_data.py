import csv
import json


from deppllama_utils import *
 
from datasets import load_dataset
import pandas as pd

#LOAD INPUT TSV files 
def load(input_file_path):
    dataset_df = pd.read_csv(input_file_path, header=None, usecols=[0,1, 2, 3], names=['0', '1', '2', '3'], sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8').astype(str)
    dataset_df = dataset_df.rename(
        columns={"0": "id", "1": "prefix", "2": "input_text", "3": "target_text"}
    )
    #dataset_df["prefix"] = ""
    dataset_df = dataset_df[["id", "input_text", "target_text", "prefix"]]
    return dataset_df

def load_and_prepare_data(input_file_path: str):

    df = load(input_file_path)

    dataset_data = [
        {
            "instruction": "Parse this sentence:",
            "input": row_dict["input_text"],
            "output": row_dict["target_text"]
        }
        for row_dict in df.to_dict(orient="records")
    ]
      

    return dataset_data

#-------------------
#    LOAD DATA 
#-------------------
def creating_data(parameters):
    train_data = load_and_prepare_data(parameters.input_train_path)
    dev_data = load_and_prepare_data(parameters.input_dev_path)

    tmp_train_file_name = "tmp_train.json"
    tmp_dev_file_name = "tmp_dev.json"
    with open(tmp_train_file_name, "w") as f:
        json.dump(train_data, f)
    with open(tmp_dev_file_name, "w") as f:
        json.dump(dev_data, f)

    json_train = load_dataset("json", data_files=tmp_train_file_name)
    json_dev = load_dataset("json", data_files=tmp_dev_file_name)
    return json_train, json_dev
