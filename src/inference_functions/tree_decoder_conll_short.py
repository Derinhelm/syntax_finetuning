class TreeDecoderConllShort:
    def __init__(self, representation_type):
        self.representation_type = representation_type

    def decode_tree(self, answer_output):
        try:
            lines = answer_output.split("\n")
            if "\t" in lines[0]:
                sep = "\t"
            else:
                sep = " "
            output = [line.strip().split(sep)
                for line in lines]
            res = [ { "id": line[0], "form": line[1],
                  "parent_id": line[2], "relation": line[3]
                } for line in output]
            return res
        except Exception as e:
            return str(e)

