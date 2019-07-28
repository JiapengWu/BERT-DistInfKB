from argparse import ArgumentParser
import os
import codecs
import pdb

from pytorch_pretrained_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# tokenizer.never_split += ['<blank>']

def get_args():
    parser = ArgumentParser(description='Distributed Inference over Knowledge Bases.')

    parser.add_argument('input_dir', type=str, default=None)
    parser.add_argument('output_dir', type=str, default=None)

    args = parser.parse_args()
    return args


def parse_train_test_file(input_fname, output_fname):
    total = 0
    preserved = 0
    with codecs.open(input_fname, "r", encoding="UTF-8") as f, codecs.open(output_fname, "w", encoding="UTF-8") as fout:
        for line in f:
            total+=1
            splitted = line.split("\t")
            context = splitted[2]
            tokenized_context = tokenizer.tokenize(context.replace('<TO_BE_PREDICTED>', '[MASK]'))
            masked_idx = tokenized_context.index("[MASK]")
            # pdb.set_trace()
            if len(tokenized_context) <= 256 and masked_idx <= 255:
                preserved += 1
                splitted[2] = " ".join(tokenized_context)
                fout.write("\t".join(splitted))
            # pdb.set_trace()
    print("{} examples are kept out of {}, {} are filtered.".format(preserved, total, total - preserved))

def parse_entity_desc_file(input_fname, output_fname):
    with codecs.open(input_fname, "r", encoding="UTF-8") as f, codecs.open(output_fname, "w", encoding="UTF-8") as fout:
        for line in f:
            # pdb.set_trace()
            splitted = line.split("\t")
            context = splitted[-1]
            tokenized_context = tokenizer.tokenize(context)
            splitted[-1] = " ".join(tokenized_context)
            fout.write("\t".join(splitted))
            fout.write("\n")

if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # for mode in "train", "valid", "test":
    # # mode = 'test'
    #     inp_dir = os.path.join(args.input_dir, "{}.txt".format(mode))
    #     out_dir = os.path.join(args.output_dir, "{}.txt".format(mode))
    #     parse_train_test_file(inp_dir, out_dir)
    parse_entity_desc_file(os.path.join(args.input_dir, "entities.txt"),
                          os.path.join(args.output_dir, "entities.txt"))