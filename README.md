

Создание датасета:
```
python3 dataset_creating_script.py -i 'src/data/conllu/UD_Russian-Taiga/ru_taiga-ud-train-a.conllu' 'src/data/conllu/UD_Russian-Taiga/ru_taiga-ud-train-b.conllu' -r grct -o 'ru_taiga-train'
```
или
```
python3 dataset_creating_script.py -i 'src/data/conllu/UD_Russian-Taiga/ru_taiga-ud-train-a.conllu' 'src/data/conllu/UD_Russian-Taiga/ru_taiga-ud-train-b.conllu' -r loct -o 'ru_taiga-train'
```
или
```
python3 dataset_creating_script.py -i 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu' -o 'ru_syntagrus-test' -r conll
```

## SynTagRus
```
python3 dataset_creating_script.py -i 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-train-a.conllu' 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-train-b.conllu' 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-train-c.conllu' -o 'ru_syntagrus-train' -r grct
```

```
python3 dataset_creating_script.py -i 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu' -o 'ru_syntagrus-dev' -r grct
```

```
python3 dataset_creating_script.py -i 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu' -o 'ru_syntagrus-test' -r grct
```

```
python3 dataset_creating_script.py -i 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-train-a.conllu' 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-train-b.conllu' 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-train-c.conllu' -o 'ru_syntagrus-train' -r lct
```

```
python3 dataset_creating_script.py -i 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu' -o 'ru_syntagrus-dev' -r lct
```

```
python3 dataset_creating_script.py -i 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu' -o 'ru_syntagrus-test' -r lct
```

```
python3 dataset_creating_script.py -i 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-train-a.conllu' 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-train-b.conllu' 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-train-c.conllu' -o 'ru_syntagrus-train' -r conll
```

```
python3 dataset_creating_script.py -i 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu' -o 'ru_syntagrus-dev' -r conll
```

```
python3 dataset_creating_script.py -i 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu' -o 'ru_syntagrus-test' -r conll
```

## Вычисление метрик
```
python3 score_functions.py --filepath '../../pred_results/Qwen06_Instruct_grct_syntagrus.json' --result_filepath '../../metrics/metrics_Qwen06_Instruct_grct_syntagrus.json'
```

```
python3 src/creating_metrics.py
```
