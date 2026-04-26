import copy

from config_parsing import parse_field, get_several_config_params
from config import DatasetConfig, DataRestrictionConfig, InferenceModelConfig
from inference_parser import create_adapter_name
from parameters import InferenceParameters

def create_inference_experiments(configs):
    root_output_dir_path = configs['root_output_dir_path']

    dataset_configs = parse_field(configs, "dataset", DatasetConfig)
    assert len(set(dc['treebank'] for dc in dataset_configs)) == len(dataset_configs)
    # Все treebank различны
    dataset_configs = {dc['treebank']: dc for dc in dataset_configs}

    parameters = InferenceParameters()
    several_param_names, s_params = get_several_config_params(
        configs["inference"], parameters)

    models = {}
    for model_config in configs['models']:
        model_name = model_config['name']
        if "peft_model_id" in model_config:
            model_config['adapter_name'] = model_name
            models[model_name] = model_config
        else:
            config_adapters = model_config['peft_group']
            peft_adapters = [(a, create_adapter_name(a))
                             for a in config_adapters]
            for peft_model_id, adapter_name in peft_adapters:
                adapter_model_dict = copy.deepcopy(model_config)
                adapter_model_dict['peft_model_id'] = peft_model_id
                adapter_model_dict['adapter_name'] = adapter_name
                models[model_name] = adapter_model_dict

    experiments = []
    for experiment_number, experiment_params in enumerate(s_params):
        assert len(experiment_params) == len(several_param_names)
        cur_parameters = copy.deepcopy(parameters)
        for param_i, param in enumerate(experiment_params):
            cur_parameters.__setattr__(several_param_names[param_i], param)
        cur_parameters.experiment_number = experiment_number

        model_config = InferenceModelConfig(
            models[cur_parameters.model_parameters.model_name], experiment_number)
        treebank_name = cur_parameters.treebank_parameters.treebank_name
        dataset_config = dataset_configs[treebank_name]
        data_restriction_config = DataRestrictionConfig(
            cur_parameters.treebank_parameters)

        experiments.append({"model_config": model_config,
            "data_restriction_config": data_restriction_config,
            "root_output_dir_path": root_output_dir_path,
            "dataset_config": dataset_config,
            "cur_parameters": cur_parameters})
    return experiments
