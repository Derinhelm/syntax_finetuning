

Создание датасета:
```
python3 dataset_creating_script.py -i 'src/data/conllu/UD_Russian-Taiga/ru_taiga-ud-train-a.conllu' 'src/data/conllu/UD_Russian-Taiga/ru_taiga-ud-train-b.conllu' -r grct -o 'ru_taiga-train'
```
или
```
python3 dataset_creating_script.py -i 'src/data/conllu/UD_Russian-Taiga/ru_taiga-ud-train-a.conllu' 'src/data/conllu/UD_Russian-Taiga/ru_taiga-ud-train-b.conllu' -r loct -o 'ru_taiga-train'
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
python3 dataset_creating_script.py -i 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-train-a.conllu' 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-train-b.conllu' 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-train-c.conllu' -o 'ru_syntagrus-train' -r loct
```

```
python3 dataset_creating_script.py -i 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu' -o 'ru_syntagrus-dev' -r loct
```

```
python3 dataset_creating_script.py -i 'src/data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu' -o 'ru_syntagrus-test' -r loct
```
