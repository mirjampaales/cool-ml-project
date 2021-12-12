# Named Entity Translation

Next two steps are implemented in `named_entity_extraction.py` script

## Named Entity Extraction

1. Gathered data from open-source data (tatoeba, wikimedia). 
2. Applied estonian named entity recognition model to the gathered data, to find sentences that contain named entities. 
* The NER model used on estonian data: https://huggingface.co/tartuNLP/EstBERT_NER

## Named Entity Data Augmentation

1. Gathered lists of person and location names
2. Replaced the named entities in the extracted sentences with the gathered names
3. Obtained augmented data

## Single-direction model finetuning

Here we finetune a baseline model on the augmented data that was generated in the previous step.

**Requirements for environment:**
```
sentencepiece==0.1.96
fairseq==0.10.2
```

We assume that there is an Estonian-English baseline single-directional model already trained. Our model was trained on open-source data from [OPUS](https://opus.nlpl.eu/) with [fairseq](https://fairseq.readthedocs.io/en/latest/) framework.

The `finetune_workflow.sh` script takes data folder (where the files are already divided into train-test-valid) as input and outputs a single-direction model trained on the input data.

**Steps of the script:**

1. Training data is tokenized by the baseline sentencepiece model.
2. Tokenized training data is input for fairseq-preprocess command which produces binarized data files into the specified binarized data folder.
3. The binarized data folder is input to fairseq-train command which finetunes the baseline models best checkpoint on that binarized data.

`finetune_workflow.sh` takes 12 arguments:

1. clean data folder (e.g. with files like train.et, train.en, test.et, test.en, valid.et, valid.et)
2. sentencepiece model directory path
3. tokenized data directory path (tokenized by trained sentencepiece model)
4. sentencepiece model prefix
5. source language (et)
6. target language (en)
7. binarized data directory path
8. model directory path
9. train file name prefix
10. validation file name prefix
11. test file name prefix
12. baseline best checkpoint path

**Running example:**

```
finetune_workflow.sh ../mt_project/clean_data/et_en/general ../mt_project/sp_models ../mt_project/tokenized_data_et_en_general SPM_et_en et en ../mt_project/data-bin_et_en_general ../mt_project/et_en_general_model train valid test ../mt_project/baseline/checkpoint_best.pt
```

* `clean_data/et-en/general` folder consists of files `train.et, train.en, test.et, test.en, valid.et, valid.et`
* estonian is the source language and english is the target language
* tokenized data folder should be created for every language pair direction and domain separately, so to avoid overwriting other tokenized data
