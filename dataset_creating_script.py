import argparse
from conllu import parse_tree, parse
import json
import random

from src.sentence_utils import simplify_relations, get_masked_id, tree2string_plain, \
    tree2string_loct, tree2string_lct, tree2string_grct

random.seed(23)

parser = argparse.ArgumentParser(description='Parsing 2011 Data Generator.')

parser.add_argument('-i','--inputs', help='Input File Paths', nargs='*')
parser.add_argument('-o','--output_name', help='Output File Name')
parser.add_argument('-d','--decodeinput', help='Decode Input File Path',default=None)
parser.add_argument('-r','--representation', help='representation', choices = ['lct', 'grct', 'loct'], default="lct")
parser.add_argument('-m','--max_length', help='Max Length', type=int, default=100000)
parser.add_argument('-n','--max_sentences', help='Max Number of Sentences', type=int, default=10000000)
parser.add_argument('-c','--corpus', help='corpus', default="")
parser.add_argument('-p','--masked_prob', help='masked_prob', type=float, default=0)
parser.add_argument('-t','--test_files', help='test function: provide gold-standard-file prediction-file', nargs=2)
parser.add_argument('--is_labeled', action="store_true", default=False)
parser.add_argument('--simple_relations', action="store_true", default=False)
parser.add_argument('--disable_tokenization', action="store_true", default=False)


args = parser.parse_args()
input_files = args.inputs
print(input_files)
output_name = args.output_name
print(output_name)
max_length = args.max_length
max_sentences = args.max_sentences
decode_input_file = args.decodeinput
corpus = args.corpus
masked_prob = args.masked_prob
test_files = args.test_files
representation = args.representation

is_labeled = args.is_labeled
disable_tokenization = args.disable_tokenization
simple_relations = args.simple_relations


assert masked_prob >= 0 and masked_prob <=1

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
            #if i != 135:
            #    continue

            masked_id = get_masked_id(len(sentences[i]), masked_prob)

            if disable_tokenization:
                str_input = sentences[i].metadata["text"]
            else:
                str_input = tree2string_plain(sentences[i], masked_id=masked_id)
                
            if representation == "lct":
                sent_res = tree2string_loct(trees[i], masked_id=masked_id).replace(" ", "")
            elif representation == "grct":
                sent_res = tree2string_grct(trees[i], masked_id=masked_id).replace(" ", "")
            else:
                sent_res = tree2string_lct(trees[i], masked_id=masked_id).replace(" ", "")

            #print(str_input, end="\n\n")
           # print(sent_res, end="\n\n")
            #print(corpus + "___" + str(i) + "\tparse\t" + str_input + "\t" + used_representation)
            res_list.append({"index": i, "input": str_input, "output": sent_res})
            
            
new_filename = f'src/data/{representation}_{output_name}.json'
print(new_filename)
with open(new_filename, 'w') as json_file:
    json.dump(res_list, json_file, indent=4)
