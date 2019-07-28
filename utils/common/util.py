from utils.common.arg_util import get_args
from torchtext.data import Field, RawField
from datasets import Dataset, EntitiesAndRelations, get_entity_description_indices, tokenizer, bert_post_processing
import os
import torch
import torch.nn.functional as F
import re


def process_args():
    args = get_args()
    args_config = args
    args_config.gpu = args.gpu if torch.cuda.is_available() else -1
    args_config.ent_correl_model_path = os.path.join("pretrained_models", args.ent_correl_model_path)
    args_config.embed_model_path = os.path.join("pretrained_models", args.embed_model_path)
    args_config.desc_model_path = os.path.join("pretrained_models", args.desc_model_path)

    args_config.dataset_train = os.path.join(args_config.dataset_dir, 'train.txt')
    args_config.dataset_valid = os.path.join(args_config.dataset_dir, 'valid.txt')
    args_config.dataset_test = os.path.join(args_config.dataset_dir, 'test.txt')
    return args_config


def preprocessing_empty(x):
    return " " if x is None else x


def calc_bilinear(hidden, h, t):
        return torch.sum(hidden(h, t), dim=-1)


def bert_tokenization(x):
    return tokenizer.tokenize(x.replace('<TO_BE_PREDICTED>', '[MASK]'))

def normalizeString(s):
    s = re.sub(r"[^A-Za-z0-9(),!?\'\"-:]", " ", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = s.strip()
    return s


def init_variables(args):

    ent_and_rels = EntitiesAndRelations(args,
                                        os.path.join(args.entities_dir, 'entities.txt'),
                                        os.path.join(args.entities_dir, 'entity2id.txt'),
                                        os.path.join(args.entities_dir, 'relation2id.txt'))
    ent_and_rels.calc_idx2name()

    def mid_postprocessing(batch, vocab, train):
        post_batch = []
        for arr in batch:
            post_arr = []
            for x in arr:
                post_arr.append(ent_and_rels.mid2idx.get(x, ent_and_rels.unk_idx))
            post_batch.append(post_arr)
        return post_batch

    tokenizer.never_split=['<blank>']

    CONTEXT_FIELD = Field(include_lengths=True, batch_first=True, fix_length=args.input_fixed_length, postprocessing=bert_post_processing,
                          pad_token='[PAD]', init_token='[CLS]', use_vocab=False)

    CANDIDATES_FIELD = Field(use_vocab=False, tokenize=(lambda s: s.split(',')), postprocessing=mid_postprocessing,
                             include_lengths=True, batch_first=True)

    distinfkb_fields = [
        ('corpus_id', RawField()),
        ('context_id', RawField()),
        ('context', CONTEXT_FIELD),
        ('candidates', CANDIDATES_FIELD),
        ('val_idx', Field(use_vocab=False, sequential=False)),
        ('val_id', CANDIDATES_FIELD),
        ('val_text', RawField()),
        ('selection', CANDIDATES_FIELD),
        ('selection_txt', None),  # RawField(preprocessing=preprocessing_empty)),
        ('selection_prob', None)]  # RawField(preprocessing=preprocessing_empty))]

    distinfkb_iters = Dataset.iters(args, 'dataset', distinfkb_fields, args.gpu, args.batch_size,
                                    args.dataset_dir,
                                    args.dataset_train, args.dataset_valid, args.dataset_test,
                                    args.load_from_file)

    prediction_idx = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

    distinfkb_train_iter, distinfkb_valid_iter, distinfkb_test_iter = distinfkb_iters
    distinfkb_train_iter.repeat = False;distinfkb_valid_iter.repeat = False;distinfkb_test_iter.repeat = False
    distinfkb_train_iter.shuffle = True;distinfkb_valid_iter.shuffle = True;distinfkb_test_iter.shuffle = True

    ent_desc_idx = get_entity_description_indices(args, ent_and_rels)
    pad_idx = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]

    return ent_and_rels, distinfkb_train_iter, distinfkb_valid_iter, distinfkb_test_iter, prediction_idx, ent_desc_idx, pad_idx, CONTEXT_FIELD, CANDIDATES_FIELD


def forward_batch(batch, model, ent_and_rels, ent_desc_idx, args,
                  prediction_idx, pad_idx, analysis_mode=False, idx2mid=None, itos=None):
    # Selection Mask
    tails_idx, tail_len = batch.selection

    # pdb.set_trace()
    tails_idx = tails_idx[:, :args.topk]
    tails_mask = tails_idx != ent_and_rels.pad_idx

    candidates_idx, candidate_len = batch.candidates
    candidates_idx[candidates_idx == -1] = ent_and_rels.unk_idx
    can_mask = candidates_idx != ent_and_rels.pad_idx
    # ent_desc_idx: n_ents * 200, 245118, 200
    # (bsz, csz, 200)
    candidates_txt = get_candidate_text_idx(ent_desc_idx, candidates_idx, args.input_fixed_length)

    # Compute Context Embedding
    # contexts: bsz, context_len
    contexts, context_lens = batch.context
    # contexts = contexts[:, :512]; candidates_txt = candidates_txt[:, :512]

    res = model(contexts, candidates_txt, (contexts == prediction_idx).nonzero()[:, 1], pad_idx)

    # res = 0
    if analysis_mode:
        ctxt_dist, betas, candidate_tail_correlation, cncpt_dist = res
    else:
        total_probs = F.log_softmax(torch.where(can_mask == 1, res, -10e20 * torch.ones_like(can_mask.float())), dim=-1)
        return res, total_probs

def get_candidate_text_idx(ent_desc_idx, candidates_idx, input_fixed_length):
    batch_size = candidates_idx.shape[0]
    return torch.gather(ent_desc_idx.expand(batch_size, -1, -1), 1,
                        candidates_idx.expand(input_fixed_length, -1, -1).transpose(0, 1).transpose(1, 2))
