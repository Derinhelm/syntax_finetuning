class TreeDecoderConllShort:
    def __init__(self, representation_type):
        self.representation_type = representation_type

    def decode_tree(self, answer_output):
        try:
            output = [line.strip().split(" ")
                for line in answer_output.split("\n")]
            res = [ { "id": line[0], "form": line[1],
                  "parent_id": line[2], "relation": line[3]
                } for line in output]
            return res
        except Exception as e:
            return str(e)

