import itertools
import re
import string

def split_merged_id_form(split_line):
# Обработка строк со "слипшимися id и form" ("5ранний")
    if (len(split_line) > 0) and split_line[0][0].isdigit() and \
            ((len(split_line) == 9 and split_line[1] == "_") or
             (len(split_line) == 10 and set(split_line[1:6]) <= {"_", "__"})): # TODO
        digits = ''.join(list(itertools.takewhile(
            lambda x: x.isdigit(), split_line[0])))
        other = split_line[0][len(digits):]
        if all([symb.isalpha() or symb in string.punctuation or symb in {"«", "»"}
                for symb in other]):
            split_line = [digits, other] + split_line[1:]
    return split_line

class TreeDecoderConll:
    def __init__(self, representation_type):
        self.representation_type = representation_type
        self.errors = []

    def decode_tree(self, answer_output, check_seq=False):
        try:
            res = []
            for line in answer_output.strip().split('\n'):
                split_line = re.split(r'\s+', line)
                split_line = [el for el in split_line if len(el) > 0]
                split_line = split_merged_id_form(split_line)
                if len(split_line) > 10 and set(split_line[2:-4]) <= {"_", "__"}:
                    split_line = split_line[:2] + ["_"] * 4 + split_line[-4:]
                    # Удаление лишних "_" в середине
                if len(split_line) == 10:
                    if not split_line[0].isdigit():
                        self.errors.append(("Not digit id", line))
                    elif split_line[1] == "_":
                        self.errors.append(("Wrong form", line)) # TODO: Более понятное описание
                    elif not split_line[6].isdigit():
                        self.errors.append(("Not digit parent_id", line))
                    elif not split_line[7].isalpha():
                        self.errors.append(("Wrong relation", line))
                    else:
                        line_res = {}
                        line_res["id"] = split_line[0] # str as in gold
                        line_res["form"] = split_line[1]
                        line_res["parent_id"] = split_line[6] # TODO: возможно, брать -3 и -4
                        line_res["relation"] = split_line[7]
                        res.append(line_res)
            return res
        except Exception as e:
            return str(e)

