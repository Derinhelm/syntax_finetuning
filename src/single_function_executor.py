from deppllama_train_qlora import conduct_experiment
from inference_parser import start_inference_experiment

class InferenceExecutor:
    def __call__(self, params):
        return start_inference_experiment(params)

class FineTuningExecutor:
    def __call__(self, params):
        print("-" * 10, params.__dict__, sep='\n')
        return conduct_experiment(params)