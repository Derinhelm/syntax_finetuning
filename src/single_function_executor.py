from inference_parser import start_inference_experiment

class InferenceExecutor:
    def __call__(self, params):
        return start_inference_experiment(params)