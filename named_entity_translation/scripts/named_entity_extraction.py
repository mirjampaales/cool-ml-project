from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline
from collections import defaultdict
import random
import argparse

def generate_person_lines(mode, first_name_f, last_name_f):
    with open(first_name_f, 'r', encoding='utf-8') as first_file, open(last_name_f, 'r', encoding='utf-8') as last_file:
        first_name_list = first_file.readlines()
        last_name_list = last_file.readlines()

    if mode == 'full':
        random.shuffle(first_name_list)
        random.shuffle(last_name_list)
        big_first_name_list = first_name_list + first_name_list + first_name_list + first_name_list + first_name_list

        full_name_list = []
        for first_name, last_name in zip(big_first_name_list[:1200], last_name_list[:1200]):
            full_name_list.append(first_name.strip() + ' ' + last_name.strip())
        return full_name_list
    else:
        return first_name_list, last_name_list

def generate_location_lines(location_file):
    with open(location_file, 'r', encoding='utf-8') as loc_f:
        place_name_list = loc_f.readlines()
    return place_name_list

def generate_from_templates(ne_mode, person_name_mode, ner_lines, new_file1, new_file2, loc_file, first_name_file, last_name_file):
    if ne_mode == 'person':
        beginning_ne = 'B-PER'
        continue_ne = 'I-PER'
        if person_name_mode == 'full':
            name_list = generate_person_lines('full', first_name_file, last_name_file)
        else:
            first_name_list, last_name_list = generate_person_lines('separate', first_name_file, last_name_file)
            name_list = first_name_list + last_name_list
    elif ne_mode == 'location':
        beginning_ne = 'B-LOC'
        continue_ne = 'I-LOC'
        name_list = generate_location_lines(loc_file)
    else:
        raise Exception('Invalid mode')

    for new_name in name_list:
        # iterating over sentences
        for lines, nes in ner_lines.items():
            new_sentence_et = lines[0].strip()
            new_sentence_en = lines[1].strip()

            name_dict = []
            name_indexes=[]
            the_entire_name=''
            # iterating over named entities in the sentence and extracting necessary info and formatting it better
            for ne in nes:
                if ne['entity'] == beginning_ne:
                    if len(name_indexes) > 0 and the_entire_name != '':
                        name_dict.append((name_indexes, the_entire_name))
                        name_indexes = []
                        the_entire_name = ''
                    if ne['word'][0:2] != '##':
                        name_indexes.append(ne['index'])
                        the_entire_name += ne['word']
                elif ne['entity'] == continue_ne:
                    if len(name_indexes) > 0 and name_indexes[-1] + 1 == ne['index']:
                        name_indexes.append(ne['index'])
                        if ne['word'][0:2] == '##':
                            the_entire_name += ne['word'][2:]
                        else:
                            the_entire_name += ' ' + ne['word']
                else:
                    if len(name_indexes) > 0 and the_entire_name != '':
                        name_dict.append((name_indexes, the_entire_name))
                        name_indexes = []
                        the_entire_name = ''

            # replacing the names in the sentences with new names
            for item in name_dict:
                if (item[1] in new_sentence_et) and (item[1] in new_sentence_en):
                    new_sentence_et = new_sentence_et.replace(item[1], new_name.strip(), 1)
                    new_sentence_en = new_sentence_en.replace(item[1], new_name.strip(), 1)

            if new_sentence_et != lines[0].strip():
                with open(new_file1 + '.' + new_name.strip(), 'a', encoding='utf-8') as new_out_et, open(new_file2 + '.' + new_name.strip(), 'a', encoding='utf-8') as new_out_en:
                    new_out_et.write(new_sentence_et+'\n')
                    new_out_en.write(new_sentence_en+'\n')


def main(args):
    tokenizer_et = BertTokenizer.from_pretrained('tartuNLP/EstBERT_NER')
    model_et = BertForTokenClassification.from_pretrained('tartuNLP/EstBERT_NER')

    nlp_et = pipeline("ner", model=model_et, tokenizer=tokenizer_et)

    et_file = args.inputfile1
    en_file = args.inputfile2

    with open(et_file, 'r', encoding='utf-8') as f1, open(en_file, 'r', encoding='utf-8') as f2:
        et_lines = f1.readlines()
        en_lines = f2.readlines()

    ner_lines = defaultdict()

    for line1, line2 in zip(et_lines, en_lines):
        ner_results1 = nlp_et(line1.strip())
        if len(ner_results1) > 0:
            ner_lines[(line1, line2)] = ner_results1

    with open(args.templatefile1, 'w', encoding='utf-8') as wiki_et_out, open(args.templatefile2, 'w', encoding='utf-8') as wiki_en_out:
        for key, value in ner_lines.items():
            if len(value) > 0:
                wiki_et_out.write(key[0])
                wiki_en_out.write(key[1])

    generate_from_templates('person', 'separate', ner_lines, args.destfile1, args.destfile2, args.locnamefile, args.firstnamefile, args.lastnamefile)
    generate_from_templates('person', 'full', ner_lines, args.destfile1, args.destfile2, args.locnamefile, args.firstnamefile, args.lastnamefile)
    generate_from_templates('location', '', ner_lines, args.destfile1, args.destfile2, args.locnamefile, args.firstnamefile, args.lastnamefile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--firstnamefile', help='first name file', required=True)
    parser.add_argument('--lastnamefile', help='last name file', required=True)
    parser.add_argument('--locnamefile', help='location name file', required=True)
    parser.add_argument('--inputfile1', help='input file path language one', required=True)
    parser.add_argument('--inputfile2', help='input file path language two', required=True)
    parser.add_argument('--templatefile1', help='template file path language one', required=True)
    parser.add_argument('--templatefile2', help='template file path language two', required=True)
    parser.add_argument('--destfile1', help='destination file path language one', required=True)
    parser.add_argument('--destfile2', help='destination file path language two', required=True)

    args = parser.parse_args()
    main(args)