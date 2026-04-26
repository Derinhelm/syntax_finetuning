from collections import OrderedDict
import itertools

def parse_field(configs, field_name, field_cls):
    if field_name in configs:
        if isinstance(configs[field_name], list):
            field_configs = [ field_cls(path_c) for path_c in configs[field_name] ]
        else:
            field_configs = [ field_cls(configs[field_name]) ]
        configs.pop(field_name)
    else:
        field_configs = []
    print(field_configs)
    return field_configs


def get_several_config_params(configs, parameters):
    several_parameters = OrderedDict()
    for param_name, param_values in configs.items():
        if isinstance(param_values, list):
            several_parameters[param_name] = param_values # Several parameters
        else:
            parameters.__setattr__(param_name, param_values) # One parameter

    several_param_names = list(several_parameters.keys())
    s_params = list(itertools.product(*several_parameters.values()))
    if not s_params:
        s_params = [{}]
    return several_param_names, s_params
