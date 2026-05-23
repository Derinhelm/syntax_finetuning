from inference_functions.vllm_creating_logit_processor import fold_bracket_seq

OP = '['
CP = ']'

class TreeDecoder:
    def __init__(self, representation_type):
        self.representation_type = representation_type

    def parseExpression(self, expression):
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

    def _parse(self, answer_output ):
        try:
            retExp, nodeMap = self.parseExpression(answer_output)
            tree = self.toTree(retExp)
            res = self.decode(tree, nodeMap)
        except Exception as err:
            return err
        else:
            return res

    def _decode(self, tree, node, nodeMap, parent, grand_parent, tid2treenodeMap, res):
        if node not in tree:
            tid = 1
            if res:
                tid = int(max(res.keys())) + 1

            grand_parent_label = "ROOT"
            if grand_parent in nodeMap:
                grand_parent_label = nodeMap[grand_parent]

            if self.representation_type == "lct":
                res[tid] = { "id": tid, "form": nodeMap[parent], "to": grand_parent_label, "toid" : grand_parent, "deprel": nodeMap[node] }
            elif self.representation_type == "grct":
                res[tid] = { "id": tid, "form": nodeMap[node], "to": grand_parent_label, "toid" : grand_parent, "deprel": nodeMap[parent] }
            else:
                raise Exception("The representation_type\t" + self.representation_type + "\t is not supported in decoding.")

            tid2treenodeMap[parent] = str(tid)

            return

        for child in tree[node]:
            self._decode(tree, child, nodeMap, node, parent, tid2treenodeMap, res)


    def toTree(self, expression):
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

    def decode(self, tree, nodeMap):
        res = dict()
        tid2treenodeMap = dict()
        #print(tree[''][0])
        self._decode(tree, "1", nodeMap, None, None, tid2treenodeMap, res)

        for i in range(1, len(res)+1):
            if res[i]["toid"] is None:
                res[i]["toid"] = '0'
            else:
                try:
                    res[i]["toid"] = tid2treenodeMap[res[i]["toid"]]
                except:
                    res[i]["toid"] = '0'

        return res


    def decode_tree(self, answer_output, check_seq=False):
        if answer_output is None:
            return "None answer"
        if self.representation_type == "grct" and check_seq:
            fold_seq = fold_bracket_seq(answer_output)
            if "E" in fold_seq:
                return f"Error level sequence. {answer_output}. {fold_seq}"
        parsing_res = self._parse(answer_output)
        if isinstance(parsing_res, Exception):
            return str(parsing_res)
        res = []
        for token in parsing_res.values():
          t =  { 'id': str(token['id']), 'form': token['form'],
                 'parent_id': token['toid'], 'relation': token['deprel'] }
          res.append(t)
        return res

