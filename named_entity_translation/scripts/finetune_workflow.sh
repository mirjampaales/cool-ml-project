#!/bin/bash

#$1 = clean data folder (files like train.et, train.en, test.et, test.en, valid.et, valid.et)
#$2 = sentencepiece model directory path
#$3 = tokenized data directory path
#$4 = sentencepiece model prefix
#$5 = source language (en/et/ru/de)
#$6 = target language (en/et/ru/de)
#$7 = binarized data directory path
#$8 = model directory path
#$9 = train file name prefix
#$10 = test file name prefix
#$11 = validation file name prefix

python /gpfs/space/home/mtars/COVID-MT_PROJECT/scripts/spm_tokenization.py --datadir $1 --spmodelpath $2 --destdir $3 --modelprefix $4 --trainprefix $9

SPM_PATH=$2
SPM_PREFIX=$4
SPM_DICT="${SPM_PATH}/${SPM_PREFIX}.vocab"
FAIRSEQ_COMPAT_DICT="${SPM_PATH}/${SPM_PREFIX}.compat_vocab"
tail -n +4 $SPM_DICT | cut -f1 | sed 's/$/ 100/g' > $FAIRSEQ_COMPAT_DICT

# preprocessing 
DATA_FOLDER=$3
SRC_LANG=$5
TGT_LANG=$6
DEST_DIR=$7
fairseq-preprocess \
    --srcdict $FAIRSEQ_COMPAT_DICT \
    --tgtdict $FAIRSEQ_COMPAT_DICT \
    --source-lang $SRC_LANG --target-lang $TGT_LANG \
    --trainpref $DATA_FOLDER/$9 \
    --validpref $DATA_FOLDER/${10} \
    --testpref $DATA_FOLDER/${11} \
    --destdir $DEST_DIR --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20
	
	
# baseline training
CHECKPOINT_DIR=$8
BEST_CP_PATH=${12}
fairseq-train --fp16 \
   $DEST_DIR \
   --source-lang $SRC_LANG --target-lang $TGT_LANG \
   --arch transformer --share-all-embeddings \
   --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
   --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
   --max-tokens 15000 --update-freq 8 \
   --finetune-from-model $BEST_CP_PATH \
   --save-interval-updates 5000 \
   --keep-interval-updates 32 \
   --save-dir $CHECKPOINT_DIR \
   --tensorboard-logdir $CHECKPOINT_DIR/log-tb
   
   
   
   



