

Создание датасета с полными синтаксическими типами связи:
```
python3 dataset_creating_script.py -c config_dataset_full_rel.yaml
```

Создание датасета с усеченными синтаксическими типами связи:
```
python3 dataset_creating_script.py -c config_dataset_simple_rel.yaml
```

## Вычисление метрик
```
python3 score_functions.py --filepath '../../pred_results/Qwen06_Instruct_grct_syntagrus.json' --result_filepath '../../metrics/metrics_Qwen06_Instruct_grct_syntagrus.json'
```

```
python3 src/creating_metrics.py
```
