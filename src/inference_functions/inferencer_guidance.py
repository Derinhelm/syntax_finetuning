from guidance import models, gen, select

class LLMInferencerGuidance:
    def __init__(self, original_model_id, peft_model_id, is_instruct, seed, model_library, max_tokens):
        self.model_library = "guidance"
        self.model = models.Transformers(original_model_id)

    def get_llm_output(self, input_text, input_tokens=None):
        print(f"input_tokens: {input_tokens}")
        prompt = f"""
Пример: Предложение <Началу работ препятствовал недостаток финансирования .> в формате CONLL:
1	Началу	3	iobj
2	работ	1	nmod
3	препятствовал	0	root
4	недостаток	3	nsubj
5	финансирования	4	nmod
6	.	3	punct
Задание: Верни в формате CONLL предложение <{" ".join(input_tokens)}>:
Результат должен состоять из {len(input_tokens)} строк в формате CONLL. Во втором столбце должны быть токены {str(list(input_tokens))}. Нельзя нарушать порядок токенов. Нельзя добавлять токены. Нельзя удалять токены.
"""
        print(prompt)
        lm = self.model # copying
        indexes = [str(i) for i in range(len(input_tokens))]
        relations = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc', 'ccomp', 'compound', 'conj', 'cop', 'csubj', '', 'dep', 'det', 'discourse', 'dislocated', 'expl', 'fixed', 'flat', 'iobj', 'list', 'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl', 'orphan', 'parataxis', 'punct', 'root', 'vocative', 'xcomp']
        print(f"relations: {relations}")
        
        for w_i, w in enumerate(input_tokens):
            lm += f"{w_i}\t{w}\t"
            print(lm)
            lm += select(indexes, name=f'ind_{w_i}')
            ind = f'ind_{w_i}'
            print(lm)
            print(f"{ind}, {lm[ind]}")
            print("\n")
            lm += "\t"
            lm += select(relations, name=f'rel_{w_i}') + "\n"
            print(lm)
            r = f'rel_{w_i}'
            print(f"{r}, {lm[r]}")
        full_output = str(lm)
        result = full_output[len(prompt):]
        token_amount = (None, None)
        print(full_output)
        return result, full_output, token_amount

