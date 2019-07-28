import os
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='BERT Distributed Inference over Knowledge Bases.')

    parser.add_argument('--config', '-c', type=str, default=None, help='JSON file with argument for the run.')
    parser.add_argument('--iterations', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--bilinear', action='store_true', dest='bilinear')

    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--ent_correl_bilinear_size', type=int, default=50)
    parser.add_argument('--context_bilinear_size', type=int, default=50)
    parser.add_argument('--beta_bilinear_size', type=int, default=10)

    parser.add_argument('--release_every', type=int, default=30)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--topk', type=int, default=25)
    parser.add_argument('--validate_every', type=int, default=3)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--dropout_ratio', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=5.0)

    parser.add_argument('--debug', action='store_true', dest='debug')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume_snapshot', type=str, default='')
    parser.add_argument('--load_from_file', action='store_true')

    # parser.add_argument('--dataset_dir', type=str, default=os.path.join(os.getcwd(), 'data/demo/'))
    parser.add_argument('--dataset_dir', type=str, default=os.path.join(os.getcwd(), 'data/'))
    parser.add_argument('--dataset_train', type=str, default="train.txt")
    parser.add_argument('--dataset_valid', type=str, default="valid.txt")
    parser.add_argument('--dataset_test', type=str, default="test.txt")

    parser.add_argument('--save_path', type=str, default=os.path.join(os.getcwd(), 'results/'))
    parser.add_argument('--patience', type=int, default=10)

    parser.add_argument('--entities_dir', type=str, default=os.path.join(os.getcwd(), 'data/'))

    parser.add_argument('--embedding_lookup', action='store_true', dest='embedding_lookup')
    parser.add_argument('--concat', action='store_true', dest='concat')
    parser.add_argument('--ent_corr_bilinear', action='store_true', dest='ent_corr_bilinear')
    parser.add_argument('--iterative_training', action='store_true', dest='iterative_training')
    parser.add_argument('--context_bilinear', action='store_true', dest='context_bilinear')
    parser.add_argument('--beta_bilinear', action='store_true', dest='beta_bilinear')
    parser.add_argument('--pretrain_epochs', type=int, default=10)
    parser.add_argument('--train_epochs', type=int, default=1)
    parser.add_argument('--burn_in_epochs', type=int, default=30)

    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--export_steps', type=int, default=0) # 0 means not storing pretrained model

    parser.add_argument('--fine_tune_context_embedding', action='store_true')
    parser.add_argument('--fine_tune_entity_embedding', action='store_true')
    parser.add_argument('--fine_tune_ent_correl_bilinear', action='store_true')

    parser.add_argument('--fine_tune_embed_model', action='store_true')
    parser.add_argument('--fine_tune_desc_model', action='store_true')

    parser.add_argument('--stem', action='store_true', dest='stem')
    parser.add_argument('--max_vocab_size', type=int, default=None)
    parser.add_argument('--name', type=str, default='no_name', help='Name of the run.')
    parser.add_argument('--server', type=str, default='localhost', help='Visdom server url.')
    parser.add_argument('--port', type=int, default=8097, help='Visdom server port.')
    parser.add_argument('--pretrained_only', action='store_true', dest='pretrained_only')

    parser.add_argument('--load_pretrained_ent_correl', action='store_true')

    parser.add_argument('--ent_correl_model_path', type=str, default='', help='Path for EntCorrel Model.')
    parser.add_argument('--embed_model_path', type=str, default='', help='Path for EntCorrel Model.')
    parser.add_argument('--desc_model_path', type=str, default='', help='Path for EntCorrel Model.')
    parser.add_argument('--embed_hidden_size', type=int, default=100)
    parser.add_argument('--desc_hidden_size', type=int, default=50)

    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=3000)
    parser.add_argument('--input_fixed_length', type=int, default=256)

    args = parser.parse_args()

    return args
