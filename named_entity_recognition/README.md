# Named Entity Recognition

Named entity recognition was tackled in this project to enable the client extract additional keywords - names - from
their existing dataset. As this data is multilingual (and the original language is not known) we decided to experiment
with multilingual NER. We chose XLM-R as our pretrained base model and finetuned both its base and large variants. As
our initial automatic language recognition experiments showed that Estonian, Finnish and German were the most common
langauges in Ajapaik's dataset, those were chosen for the multilingual model as well.

## Finetuning

The workflow of data preparation and model training can be found in the `finetune_xlmr.ipynb` notebook. This consists of
unifying the datasets across languages, setting up the training environment and training the models. The actual
experiments were performed on the UT HPC cluster using Nvidia Tesla V100 GPUs with 32 GB of VRAM.

We used the following standard NER datasets:

- English - CoNLL 2003 shared task dataset for English (Sang, et al.)
- Estonian - Estonian NER dataset (Tkachencko, et al.)
- Finnish - Turku NER Corpus (Luoma, et al.)

Because the datasets of different languages contain a different annotations, we removed all labels except for person
(PER), location (LOC) and organization (ORG) as those were present in all langauges.

The finetuning process was based on the [`xlm-roberta-ner`](https://github.com/mukhal/xlm-roberta-ner) repository.

## Results:

The table below demonstrates the test set F1 scores of all models trained as a part of this project and how they compare
to the current state-of-the-art. Detailed results for test and validation results for each model sets can be found in
the `finetuned_models` directory.

|                            | EN     | ET      | FI       |
|----------------------------|--------|---------|----------|
| XLM-R (base)               | 0.9221 | 0.8408  | 0.8892   |
| XLM-R (large)              | 0.9285 | 0.8513  | -        |
| XLM-R (multilingual base)  | 0.9225 | 0.8400  | 0.8791   |
| XLM-R (multilingual large) | 0.9228 | 0.8295  | 0.8871   |
| SOTA                       | 0.946  | 0.8986  | 0.9209   |

Although our scores don't match the SOTA, we see that the multilinguality does not affect the performance significantly 
in most cases. Therefore, in a real world use case, a multilingual NER model can be considered and applying the 
techniques from the SOTA papers on multilingual models could have potential as well.

Some things that should be noted about these scores:

- We only used a limited set of annotation labels, which were common for all languages whereas the SOTA models were
  trained on all labels. Therefore, these are not entirely comparable.
- We were unable to verify that the Estonian NER dataset (and its test set) is the same version as used in the SOTA
  experiments.
- For unknown reasons, we were unable to successfully finetune XLM-R large to Finnish NER (F1 score was always 0).

The SOTA result info was taken from the following papers:

- English: https://aclanthology.org/2021.acl-long.206.pdf
- Finnish: http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.567.pdf
- Estonian: https://aclanthology.org/2021.nodalida-main.2.pdf

## Applying on Ajapaik's dataset

TODO