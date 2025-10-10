import argparse
from conllu import parse_tree, parse
import json
import random
import traceback

import pandas as pd


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

def print_tree(node, tab = ""):
    print(tab + str(node))

    for child in node.children:
        print_tree(child, tab=tab+"\t")

def simplify_relations(node):
    node.token["deprel"] = node.token["deprel"].split(":")[0]
    for child in node.children:
        simplify_relations(child)


def get_masked_id(N, prob):
    lista = []
    for num in range(1, N+1):
        if random.random() <= prob:
            lista.append(num)
    return lista

def tree2string_plain(tokenlist, mylist = [], masked_id=[]):
    res = ""
    for token in tokenlist:
        if isinstance(token["id"], int):
            if token["id"] in masked_id:
                res += "<mask> " 
            else:
                res += token["form"] + " "
    return res.rstrip()

# LOCT

def tree2list_loct(node, mylist = [], masked_id=[]):

        if node.token["id"] in masked_id:
            node_str = "<mask>"
        else:
            node_str = node.token["form"].replace("[", "-LRB-").replace("]", "-RRB-")

        mylist.append(OP + " " + node_str)
        for child in node.children:
            tree2list_loct(child, mylist=mylist, masked_id=masked_id)
        mylist.append(CP)


def tree2string_loct(node, masked_id=[]):
    my_list = []
    tree2list_loct(node, my_list, masked_id)
    return " ".join(my_list)


# LCT

def tree2list_lct(node, mylist = [], masked_id=[]):
    if node.token["id"] in masked_id:
        node_str = "<mask>"
    else:
        node_str = node.token["form"].replace("[", "-LRB-").replace("]", "-RRB-")

    mylist.append(OP + " " + node_str)
    is_deprel_written = False

    if len(node.children) == 0:
        mylist.append(OP + " " + node.token["deprel"] + " " + CP)

    for i, child in enumerate(node.children):        
        if child.token["id"] > node.token["id"] and not is_deprel_written:
                mylist.append(OP + " " + node.token["deprel"] + " " + CP)
                is_deprel_written = True

        tree2list_lct(child, mylist=mylist, masked_id=masked_id)

        if i == len(node.children) - 1 and not is_deprel_written:
                mylist.append(OP + " " + node.token["deprel"] + " " + CP)
                is_deprel_written = True            

    mylist.append(CP)

def tree2string_lct(node, masked_id=[]):
    my_list = []
    tree2list_lct(node, my_list, masked_id)
    return " ".join(my_list)

# GRCT

def tree2list_grct(node, mylist = [], masked_id=[]):
    node_str = node.token["deprel"]

    mylist.append(OP + " " + node_str)
    is_deprel_written = False

    if len(node.children) == 0:
        if node.token["id"] in masked_id:
            node_str = "<mask>"
        else:
            node_str = node.token["form"].replace("[", "-LRB-").replace("]", "-RRB-")
        mylist.append(OP + " " + node_str + " " + CP)

    for i, child in enumerate(node.children):        
        if child.token["id"] > node.token["id"] and not is_deprel_written:
            if node.token["id"] in masked_id:
                node_str = "<mask>"
            else:
                node_str = node.token["form"].replace("[", "-LRB-").replace("]", "-RRB-")

            mylist.append(OP + " " + node_str + " " + CP)
            is_deprel_written = True

        tree2list_grct(child, mylist=mylist, masked_id=masked_id)

        if i == len(node.children) - 1 and not is_deprel_written:
            if node.token["id"] in masked_id:
                node_str = "<mask>"
            else:
                node_str = node.token["form"].replace("[", "-LRB-").replace("]", "-RRB-")
            
            mylist.append(OP + " " + node_str + " " + CP)
            is_deprel_written = True            

    mylist.append(CP)

def tree2string_grct(node, masked_id=[]):
    my_list = []
    #print_tree(node)
    tree2list_grct(node, my_list, masked_id)
    return " ".join(my_list)


#file = open(input_file, "r", encoding="utf8")
#content = file.readlines()


#=================DECODER=======================

# These are general decoder methods

def toTree(expression):
    tree = dict()
    msg =""
    stack = list()
    for char in expression:
        if(char == OP):
            stack.append(msg)
            msg = ""
        elif char == CP:
            parent = stack.pop()
            if parent not in tree:
                tree[parent] = list()
            tree[parent].append(msg)
            msg = parent
        else:
            msg += char
    return tree

def parseExpression(expression):
    nodeMap = dict()
    counter = 1
    node = ""
    retExp =""
    for char in expression:
        if char == OP or char == CP :
            if (len(node) > 0):
                nodeMap[str(counter)] = node;
                retExp += str(counter)
                counter +=1
            retExp += char
            node =""
        elif char == ' ': continue
        else :
            node += char
    return retExp,nodeMap


def printTree(tree, node, nodeMap):
    if node not in tree:
        return 
    print('%s -> %s' % (nodeMap[node], ' '.join(nodeMap[child] for child in tree[node]))) 
    for child in tree[node]:
        printTree(tree, child, nodeMap)

def _decode(tree, representation_type, node, nodeMap, parent, grand_parent, tid2treenodeMap, res):
    if node not in tree:
        tid = 1
        if res:
            tid = int(max(res.keys())) + 1

        grand_parent_label = "ROOT"
        if grand_parent in nodeMap:
            grand_parent_label = nodeMap[grand_parent]

        if representation_type == "lct":
            res[tid] = { "id": tid, "form": nodeMap[parent], "to": grand_parent_label, "toid" : grand_parent, "deprel": nodeMap[node] }
        elif representation_type == "grct":
            res[tid] = { "id": tid, "form": nodeMap[node], "to": grand_parent_label, "toid" : grand_parent, "deprel": nodeMap[parent] }
        else:
            raise Exception("The representation_type\t" + representation_type + "\t is not supported in decoding.")
        
        if VERBOSE:
            print(res[tid])

        tid2treenodeMap[parent] = str(tid)
        
        return 
    
    for child in tree[node]:
        _decode(tree, representation_type, child, nodeMap, node, parent, tid2treenodeMap, res) 

def decode(tree, nodeMap, representation_type="lct"):
    res = dict()
    tid2treenodeMap = dict()
    #print(tree[''][0])
    _decode(tree, representation_type, "1", nodeMap, None, None, tid2treenodeMap, res)

    for i in range(1, len(res)+1):
        if res[i]["toid"] is None:
            res[i]["toid"] = 0
        else:
            try:
                res[i]["toid"] = tid2treenodeMap[res[i]["toid"]]
            except:
                res[i]["toid"] = 0

    return res

# END of general purpose decoder methods

def check_inconsistencies(gold_tree, pred_tree):
    if len(gold_tree) != len(pred_tree):
        if abs(len(gold_tree) - len(pred_tree)) > 5:
            raise Exception("There is something strange: the difference of the trees in terms of length is too much.")
        if len(pred_tree) > len(gold_tree):
            raise Exception("There is something strange: the predicted sentence should not be longer than the original one.")    

        return True

    for i in range(1, len(gold_tree)+1):
        if gold_tree[i]["form"] != pred_tree[i]["form"]:
            #print(gold_tree[i]["form"])
            return True
    return False


def solve_inconsistencies(gold_tree, pred_tree):
    res = dict()

    if VERBOSE:
        print(gold_tree)
        print()
        print(pred_tree)

    new_id_map = dict()
    old_ids_cache = dict()

    for i in range(1, min(len(gold_tree)+1, len(pred_tree)+1)):
        old_id = pred_tree[i]["id"]
        old_toid = int(pred_tree[i]["toid"])
        old_ids_cache[i] = old_toid

        if VERBOSE:
            print("ID\t" + str(i) + "\t" + str(pred_tree[i]["form"]))
        
        if gold_tree[i]["form"] == pred_tree[i]["form"]:
            res[i] = pred_tree[i]

            new_id_map[old_id] = old_id
        else:
            if VERBOSE:
                print("Problems\t" + str(i))
            found_token = False

            #find the word in the gold sentence
            for v in range(1, 200):
                for sign in [1, -1]:
                    j = v * sign

                    if (i + j) >= 1 and (i + j) <= len(gold_tree) and gold_tree[i + j]["form"] == pred_tree[i]["form"]:
                        old_id = pred_tree[i]["id"]
                        old_form = pred_tree[i]["form"]
                        old_rel = pred_tree[i]["deprel"]

                        if old_toid < 0:
                            found_token = False                        
                            break

                        if old_toid == 0: #root
                            old_to_form = "root"
                        else:
                            old_to_form = pred_tree[int(old_toid)]["form"]
                        
                        new_id = gold_tree[i + j]["id"]
                        if VERBOSE:
                            print("ID_CHANGED\t" + str(old_id) + "\t" + str(new_id))
                        found_token = True                        
                        break
                if found_token:
                    break

            if found_token:
                #old_toid will be changed later
                new_token = { "id": new_id, "form": old_form, "to": gold_tree[new_id]["form"], "toid" : int(old_toid), "deprel": old_rel }
                res[new_id] = new_token
                new_id_map[old_id] = new_id

    # ADD MISSING TOKENS
    for i in range(1, len(gold_tree)+1):
        if i not in res:
            new_deprel = "UNK"
            if gold_tree[i]["form"] == "," or gold_tree[i]["form"] == "\"":
                new_deprel = "punct"
            res[i] = { "id": i, "form": gold_tree[i]["form"], "toid" : -1, "deprel": new_deprel }   

        if VERBOSE:
            print("Check" + str(i) + "\tfrom\t" + str(old_toid))
        old_toid = int(res[i]["toid"])
        new_toid = 0 
        if old_toid in new_id_map:
            new_toid = new_id_map[old_toid]
            if VERBOSE:
                print("Swapping" + str(i) + "\tfrom\t" + str(old_toid) + "\t"+ str(new_toid))
            res[i]["toid"] = new_toid
        else:
            if res[i]["deprel"] == "root":
                res[i]["toid"] = 0
            elif i in old_ids_cache:
                res[i]["toid"] = old_ids_cache[i] 
            else:
                res[i]["toid"] = 0
    if VERBOSE:
        print("new_id_map\t" + str(new_id_map))

    return res

#------------


# LCT

def get_decode(tree_string, representation_type):
    dep_array = []

    tree_string2, nodeMap = parseExpression(tree_string)
    tree = toTree(tree_string2)

    res = decode(tree, nodeMap, representation_type)

    return res

def print_line(line):
    return str(line["id"])+"\t"+line["form"]+"\t_\t_\t_\t_\t"+str(line["toid"])+"\t"+ line["deprel"]+"\t_\t_"


def print_decode(tree_string, representation_type):
    dep_array = get_decode(tree_string, representation_type)

    for i in range(1, len(res)+1):
        dep_array.append(print_line(res[i+1]))

    for i in range(len(dep_array)):
        print(dep_array[i])
    print()



#=================MAIN=======================        

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
