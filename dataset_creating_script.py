import argparse
from conllu import parse_tree, parse
import json
import random

from src.sentence_utils import simplify_relations, tree2string_plain, \
    tree2string_loct, tree2string_lct, tree2string_grct

random.seed(23)

parser = argparse.ArgumentParser(description='Transform a dataset sample')

parser.add_argument('-i','--inputs', help='Input File Paths', nargs='*')
parser.add_argument('-o','--output_name', help='Output File Name')
parser.add_argument('-r','--representation', help='representation', choices = ['lct', 'grct', 'loct'], default="lct")
parser.add_argument('-m','--max_length', help='Max Length', type=int, default=100000)
parser.add_argument('-n','--max_sentences', help='Max Number of Sentences', type=int, default=10000000)
parser.add_argument('--simple_relations', action="store_true", default=False)

args = parser.parse_args()
input_files = args.inputs
print(input_files)
output_name = args.output_name
print(output_name)
max_length = args.max_length
max_sentences = args.max_sentences
representation = args.representation

simple_relations = args.simple_relations



OP = '['
CP = ']'

VERBOSE=False

res_list = []
for input_file in input_files:
    with open(input_file, 'r') as file:
        content = file.read()

        trees = parse_tree(content)

        if simple_relations:
            for tree in trees:
                simplify_relations(tree)

        sentences = parse(content)

        for i in range( min(len(trees), max_sentences)):
            str_input = tree2string_plain(sentences[i])
                
            if representation == "lct":
                sent_res = tree2string_loct(trees[i]).replace(" ", "")
            elif representation == "grct":
                sent_res = tree2string_grct(trees[i]).replace(" ", "")
            else:
                sent_res = tree2string_lct(trees[i]).replace(" ", "")
            res_list.append({"index": i, "input": str_input, "output": sent_res})
           
new_filename = f'src/data/{representation}_{output_name}.json'
print(new_filename)
with open(new_filename, 'w') as json_file:
    json.dump(res_list, json_file, indent=4)
