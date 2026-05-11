import copy

from config_parsing import parse_field, get_several_config_params
from config import DatasetConfig, DataRestrictionConfig, InferenceModelConfig
from parameters import InferenceParameters

def create_inference_experiments(configs,
        inf_models, datasets):
    root_output_dir_path = configs['root_output_dir_path']

    if "inference" not in configs:
        return []

    parameters = InferenceParameters()
    if isinstance(configs["inference"], list):
        inference_list = configs["inference"]
    else:
        inference_list = [configs["inference"]]

    experiments = []
    for inference_params in inference_list:
      several_param_names, s_params = get_several_config_params(
        inference_params, parameters)
      for experiment_number, experiment_params in enumerate(s_params):
        assert len(experiment_params) == len(several_param_names)
        cur_parameters = copy.deepcopy(parameters)
        for param_i, param in enumerate(experiment_params):
            cur_parameters.__setattr__(several_param_names[param_i], param)
        cur_parameters.experiment_number = experiment_number

        model_config = InferenceModelConfig(
            inf_models[cur_parameters.model_parameters['model_name']],
            experiment_number)
            

        if cur_parameters.treebank_parameters['treebank_name'] is not None:
            treebank_name = cur_parameters.treebank_parameters['treebank_name']
            dataset_config = datasets[treebank_name]
        else:
            dataset_config = None # Будет использоваться из fine-tuning
        data_restriction_config = DataRestrictionConfig(
            cur_parameters.treebank_parameters)

        experiments.append({"model_config": model_config,
            "data_restriction_config": data_restriction_config,
            "root_output_dir_path": root_output_dir_path,
            "dataset_config": dataset_config,
            "cur_parameters": cur_parameters})
    return experiments
