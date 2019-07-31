from torchtext import data
import os
import torch
import hashlib
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

stopwords = set(stopwords.words("english"))

from pytorch_pretrained_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def bert_post_processing(batch, vocab, train):
    # import pdb; pdb.set_trace()
    return list(map(lambda text: tokenizer.convert_tokens_to_ids(text), batch))


class EntitiesAndRelations(object):
    def __init__(self, config, entities_txt, entity2id, rel2id):

        self.config = config

        self.mids = []
        self.mid2idx = {}
        self.mid2name = {}
        with open(entity2id) as f:
            count = f.readline().strip()
            for i, line in enumerate(f):
                mid, idx = line.strip().split('\t')
                self.mid2idx[mid] = i

        with open(entities_txt) as f:
            for line in f:
                items = line.split('\t')
                mid, _, _, _, name = items[:5]
                self.mids.append(mid)
                self.mid2name[mid]=name

        self.pad_idx = len(self.mid2idx)
        self.mids.append('<pad>')
        self.mid2idx['<pad>'] = self.pad_idx
        self.mid2name['<pad>'] = '<pad>'

        self.unk_idx = len(self.mid2idx)
        self.mids.append('<unk>')
        self.mid2idx['<unk>'] = self.unk_idx
        self.mid2name['<unk>'] = '<unk>'

        self.rels = []
        self.rel2idx = {}
        with open(rel2id) as f:
            count = f.readline().strip()
            for line in f:
                rel, idx = line.strip().split('\t')
                self.rels.append(rel)
                self.rel2idx[rel] = int(idx)

    def calc_idx2name(self):
        self.idx2name = {self.mid2idx[mid]: self.mid2name[mid] for mid in self.mids}


class BertEntityDescDataset(data.TabularDataset):
    def __init__(self, config, ent_and_rels, entities_txt_path):

        self.DESC_FIELD = data.Field(batch_first=True, fix_length=config.input_fixed_length, postprocessing=bert_post_processing,
                                     pad_token='[PAD]', init_token='[CLS]', use_vocab=False)
        self.config = config

        def mid_postprocessing(batch):
            post_batch = []
            for val in batch:
                post_batch.append(ent_and_rels.mid2idx.get(val, ent_and_rels.unk_idx))
            return post_batch

        fields = [
                ('candidates', data.RawField(postprocessing=mid_postprocessing)),
                ('guid', None),
                ('anchor_text', None),
                ('url', None),
                ('fb_name', None),
                ('context', self.DESC_FIELD)
                ]
        super(BertEntityDescDataset, self).__init__(entities_txt_path, 'tsv', fields=fields)

    def iter(self, batch_size):
        return data.BucketIterator(dataset=self, batch_size=batch_size, shuffle=True,
                        sort_key=lambda x : len(x.context), sort=False, device=self.config.gpu)


class Dataset(data.TabularDataset):
    def __init__(self, config, dataset, fields):
        self.config = config
        super(Dataset, self).__init__(dataset, 'tsv', fields=fields)

    @classmethod
    def splits(cls, config, fields, dataset_dir=None, dataset_train=None,
                dataset_valid=None, dataset_test=None, **kwargs):
        # import pdb; pdb.set_trace()
        return (cls(config, dataset_train, fields=fields, **kwargs),
                cls(config, dataset_valid, fields=fields, **kwargs),
                cls(config, dataset_test,  fields=fields, **kwargs))

    @classmethod
    def iters(cls, config, dataset_name, fields, gpu, batch_size, dataset_dir,
                dataset_train=None, dataset_valid=None, dataset_test=None, load_from_file=False, **kwargs):
        suffix = hashlib.md5('{}-{}-{}-{}'.format(dataset_name, dataset_train, dataset_valid, dataset_test).encode()).hexdigest()
        examples_path = os.path.join(dataset_dir, dataset_name + ('-stemmed' if config.stem else '') + '-{}.pkl'.format(suffix))

        save_iters = False

        # load a list of example, each example has the attributes as named in the fields

        # #TODO:debug only
        # load_from_file = True
        # save_iters = True
        if not load_from_file:
            try:
                examples = torch.load(examples_path)
            except:
                load_from_file = True
                save_iters = True

        if load_from_file:
            iters = cls.splits(config, fields, dataset_dir, dataset_train,
                                dataset_valid, dataset_test, **kwargs)
            if save_iters:
                torch.save([it.examples for it in iters], examples_path)

        if not load_from_file:
            iters = [data.Dataset(ex, fields) for ex in examples]

        # three Dataset objects
        train, valid, test = iters

        # return three iterators
        return data.BucketIterator.splits((train, valid, test), batch_size=batch_size, shuffle=True,
                repeat=False, sort=False, device=gpu, sort_key= lambda x : len(x.context))


def get_entity_description_indices(config, ent_and_rels, vocab=None):
    # vocab: CONTEXT_FIELD.vocab
    ent_inits_path = os.path.join(config.entities_dir,'ent_inits-bert.vec')
    try:
        ent_desc_idx = torch.load(ent_inits_path)
    except:
        entities_desc = BertEntityDescDataset(config, ent_and_rels, os.path.join(config.entities_dir, 'entities.txt'))
        entity_iter = entities_desc.iter(len(ent_and_rels.mids)/10)
        entity_iter.repeat = False

        ent_desc_idx = torch.LongTensor(len(ent_and_rels.mid2idx), config.input_fixed_length).zero_()
        if config.gpu > -1:
            ent_desc_idx = ent_desc_idx.cuda(config.gpu)
        # print(len(entity_iter))

        for batch in entity_iter:
            ent_desc_idx[batch.candidates] = batch.context.data

        torch.save(ent_desc_idx, ent_inits_path)
    return ent_desc_idx
