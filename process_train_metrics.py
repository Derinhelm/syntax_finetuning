import argparse
from collections import defaultdict, Counter
import json
from math import floor
import os

import matplotlib.pyplot as plt

def process_data(filename): 
    with open(filename, 'r') as f:
        data = f.read().strip()
    
    data = data.split('Training time') # Splitting by experiments
    data = [ exp.split("\n") for exp in data]
    data = [[ d for d in exp if len(d) > 0 and \
              d[0] == '{' and d[-1] == '}' and "'epoch'" in d ] for exp in data] # Deleting non-loss strings
    data = [ exp for exp in data if len(data) > 0]
    
    data = [[ json.loads(d.replace("'", "\"").replace("nan", "\"nan\"").replace("inf", "\"inf\""))
              for d in exp ] for exp in data]
    
    train_dict = {}
    eval_dict = {} 
    for exp_i, exp in enumerate(data):
        train_dict[exp_i] = defaultdict(list)
        eval_dict[exp_i] = []
        for d in exp:
            if 'loss' in d:
                train_dict[exp_i][floor(d['epoch'])].append(d)
            elif 'eval_loss' in d:
                eval_dict[exp_i].append(d)
            #else:
                #print(d)
    return train_dict, eval_dict
    
parser = argparse.ArgumentParser()
parser.add_argument("--filenames", nargs='*')
parser.add_argument("--output_dir", nargs='?', default=".")
parser_args = parser.parse_args()
filenames = parser_args.filenames
output_dir = parser_args.output_dir
    
train_dict, dev_dict = {}, {}
for filename_i, filename in enumerate(filenames):
    train_dict[filename_i], dev_dict[filename_i] = process_data(filename)
    
mean_train_dict = {}
for filename_i in train_dict:
    mean_train_dict[filename_i] = {}
    for exp_i, exp_dict in train_dict[filename_i].items():
        mean_train_dict[filename_i][exp_i] = {}
        prev_epoch = -1
        cur_values = []
        for e_i in exp_dict:
          for d in exp_dict[e_i]:
            if prev_epoch != d['epoch']:
                if prev_epoch != -1:
                    mean_train_dict[filename_i][exp_i][prev_epoch] = sum(cur_values) / len(cur_values)
                prev_epoch = d['epoch']
                cur_values = [d['loss']]
        if cur_values:
            mean_train_dict[filename_i][exp_i][prev_epoch] = sum(cur_values) / len(cur_values)
            
# Количество eval_loss
print("Количество eval_loss")
for filename_i in dev_dict:
    print(filename_i, len(dev_dict[filename_i][0]))
    
# Наличие inf, nan в loss
print("Наличие inf, nan в loss")
for filename_i in train_dict:
    for exp_i in train_dict[filename_i]:
        for epoche_i in train_dict[filename_i][exp_i]:
            wrong_loss = [d['loss'] for d in train_dict[filename_i][exp_i][epoche_i] if isinstance(d['loss'], str)]
            if wrong_loss:
                print(filename_i, exp_i, epoche_i, wrong_loss)
                
print("=" * 10)

# Наличие inf, nan в grad_norm
print("Наличие inf, nan в grad_norm")
for filename_i in train_dict:
    for exp_i in train_dict[filename_i]:
        for epoche_i in train_dict[filename_i][exp_i]:
            str_list = [d['grad_norm'] for d in train_dict[filename_i][exp_i][epoche_i] if isinstance(d['grad_norm'], str)]
            if str_list:
                print(filename_i, filenames[filename_i], "\n", exp_i, epoche_i, Counter(str_list), "\n")
        #print("-" * 10)
    #print("=" * 10)

# По всем train-данным
for filename_i, file_train in mean_train_dict.items():
    filename_dir = filenames[filename_i].split('/')[-1].split('.')[-2]
    os.mkdir(f'{output_dir}/{filename_dir}')
    for exp_i, exp_dict in file_train.items():
        expir_name = f"filename: {filename_dir}, exp: {exp_i}"
        #print(expir_name)
        prev_point_amount = 0
        if exp_dict:
            plt.plot(exp_dict.keys(), exp_dict.values())
            plt.plot([d['epoch'] for d in dev_dict[filename_i][exp_i]],
                                    [d['eval_loss'] for d in dev_dict[filename_i][exp_i]], color='red')

            for e_i in range(6):
                plt.axvline(e_i, color='grey', alpha=0.4)

            plt.title(expir_name)
            plt.savefig(f'{output_dir}/{filename_dir}/{filename_dir}_exp_{exp_i}.jpg', bbox_inches='tight')
            plt.cla()
print(f"Результаты сохранены в {output_dir}")

