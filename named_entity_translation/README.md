# Named Entity Translation

Next two steps are implemented in the data augmentation scripts in the `scripts` folder.

## Named Entity Extraction

1. Gathered data from open-source data (tatoeba, wikimedia). 
2. Applied estonian named entity recognition model to the gathered data, to find sentences that contain named entities. 
* The NER model used on estonian data: https://huggingface.co/tartuNLP/EstBERT_NER

## Named Entity Data Augmentation

1. Gathered lists of person and location names.
2. Replaced the named entities in the extracted sentences with the gathered names.
3. Obtained augmented data.

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

## Results

We performed multiple experiments. We uploaded two of the trained models: [person name model](https://drive.google.com/file/d/1g29QjyxlrWU1RM2b4JG_ukzPk2x443RF/view?usp=sharing) and [location name model](https://drive.google.com/file/d/1YQgJg-f2Ok7Aqo-EgeaPX-WRr6AeLM5b/view?usp=sharing).

**BLEU scores on visitestonia test data:**

| Model        | Result (BLEU) |
| ------------ | ------------- |
| Baseline     |    32.72      |
| Extracted_FT |    32.68      |
| Person_FT    |    16.05      |
| Location_FT  |    24.74      |
| All_FT       |    13.05      |

**Some examples of test on Ajapaik's data:**

```
Source: Karatuzi pargis, Koidu Treimann ja Aili Karp
Baseline: Karatuz Park, Dawn Treimann and Aili Karp
Person FT: Karatuz Park, Koidu Treimann and Aili Karp
```
```
Source: Saapaküla koorejaam
Baseline: Boot Village Bark Station
Location FT: Saapaküla shell station
```
```
Source: Foto.  Võru, puitelamu Lenini tn.6( Jüri t)  1984.a.
Baseline: Photo. Collage, wooden house Lenini tn.6(Jüri t) in 1984.
Location_FT: Photo. Võru, a Puitelamu Lenini tn.6( Jüri t) in 1984.
```
```
Source: Arnold Lõhmus aias viiulit mängimas
Baseline: Arnold Fragrance playing violin in the garden
Person_FT: Arnold Lõhmus playing fiddle in the garden
```
```
Source: Foto  "Liivaskulptuur "Võru vähk"  autor Kalle Pruuden Eesti juulis 2008
Baseline: Kalle Pruuden, author of the photo "Sand sculpture "Cancer of the Strangle" in July 2008
Person_FT: Photo "Liivaskulptuur "Võru cancer" by Kalle Pruuden in July 2008
```
```
Source: Perekond Kullerkup Tõusu (Palvemaja) 8 maja uksel. Rakvere Tõusu t 8
Baseline: Family Courier Rise (Prayer House) 8 at the door of the house. Rakvere Rise t 8
Location_FT: The family Kullerkup Tõusu (Palvemaja) 8 at the door of the house. Rakvere Tõusu t 8
```
```
Source: Koguva külavahe
Baseline: The Gathering Village
Person_FT: Koguva guest house (PER)
```

**How to use the finetuned models:**

Running example:

```
fairseq-interactive $DATA_DIR \
    --task translation \
    --source-lang $SRC_LANG \
    --target-lang $TGT_LANG \
    --bpe sentencepiece \
    --remove-bpe \
    --sentencepiece-model $SP_MODEL_PATH \
    --path $MODEL_DIR \
```
where 
* `$DATA_DIR` is directory with dict.\*.txt files
* `$SRC_LANG` and `$TGT_LANG` are et and et respectively
* `$SP_MODEL_PATH` is the path of sentencepiece model file (sp-model.model)
* `$MODEL_DIR` is the directory where the finetuned model checkpoint_best.pt is
