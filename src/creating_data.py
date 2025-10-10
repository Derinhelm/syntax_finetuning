from datasets import load_dataset

#-------------------
#    LOAD DATA 
#-------------------
def creating_data(parameters):
    json_train = load_dataset("json", data_files=parameters.train_file_path)
    json_dev = load_dataset("json", data_files=parameters.dev_file_path)
    return json_train, json_dev
