#!/usr/bin/env python

import sentencepiece as spm
import os
import argparse

def sp_tokenization(infile, outfile, spmodel):
    with open(infile, 'r', encoding='utf-8') as in_file:
        with open(outfile, 'w', encoding='utf-8') as out_file:
            for line in in_file:
                new_line = spmodel.encode_as_pieces(line)
                out_file.write(' '.join(new_line) + '\n')
                
def main(args):
    dir_path = args.datadir
    dest_path = args.destdir

    # creating a list of train files
    list_of_train_files = []
    for file in os.listdir(dir_path):
        if os.path.splitext(os.path.basename(file))[0] == args.trainprefix:
            list_of_train_files.append(os.path.join(dir_path, file))
    train_files = ','.join(list_of_train_files)

    # loading sentencepiece model
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(args.spmodelpath, args.modelprefix) + '.model')

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # tokenization of files with the trained sentencepiece model
    for file in os.listdir(dir_path):
        filename = os.path.basename(file)
        dest_file = os.path.join(dest_path, filename)
        sp_tokenization(os.path.join(dir_path, file), dest_file, sp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir', help='directory with files with source and target languages as extensions', required=True)
    parser.add_argument('-s', '--spmodelpath', help='sentencepiece model path', required=True)
    parser.add_argument('-t', '--destdir', help='destination directory for tokenized files', required=True)
    parser.add_argument('-m', '--modelprefix', help='sentencepiece model prefix', required=True)
    parser.add_argument('-p', '--trainprefix', help='training data file name (file prefix)', required=True)

    args = parser.parse_args()
    main(args)