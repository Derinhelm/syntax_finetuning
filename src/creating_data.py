from datasets import load_dataset

#-------------------
#    LOAD DATA 
#-------------------
def creating_data(parameters):
    json_train = load_dataset("json", data_files=parameters.input_train_path)
    json_dev = load_dataset("json", data_files=parameters.input_dev_path)
    return json_train, json_dev
