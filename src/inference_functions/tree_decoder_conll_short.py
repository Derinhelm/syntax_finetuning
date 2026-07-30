import re

class TreeDecoderConllShort:
    def __init__(self, representation_type):
        self.representation_type = representation_type

    def decode_tree(self, answer_output, check_seq=False):
        try:
            lines = answer_output.strip().split("\n")
            output = [re.split(r'\s+', line.strip())
                for line in lines]
            output = [line for line in output if len(output) == 4]
            res = [ { "id": line[0], "form": line[1],
                  "parent_id": line[2], "relation": line[3]
                } for line in output] # TODO: не рассматриваются случаи, когда form - из несколько слов (не очень хороший, но может быть)
            return res
        except Exception as e:
            return str(e)

