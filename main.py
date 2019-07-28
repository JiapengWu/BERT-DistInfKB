
import os
import time
import sys
import shutil
import json
import logging
from models.BertModel import *
import pdb
from pytorch_transformers import AdamW, WarmupLinearSchedule

from torchnet.logger.visdomlogger import VisdomSaver, VisdomTextLogger, VisdomPlotLogger
from torchnet.meter import AverageValueMeter
from utils.common.util import *

def get_filename(model_name, suffix):
    return '%s_%s_%s.pt' % (model_name, suffix,
                            time.strftime('%Y-%m-%d:%H:%M'))


def checkpoint(model, model_filename):
    snapshot_path = os.path.join(checkpoint_dir, model_filename)
    torch.save(model, snapshot_path)
    return snapshot_path


def forward_epoch(model, dataset_iter, epoch=0, mode="train"):
    if not args.debug:
        meter_acc.reset()
        meter_loss.reset()

    epoch_loss = 0; counter = 0; epoch_correct = 0
    dataset_iter.init_epoch()
    total_batch = len(dataset_iter)
    for batch_idx, batch in enumerate(dataset_iter):
        # pdb.set_trace()
        # batch = tuple(t.to(args_config.device) for t in batch)
        contexts = batch.context[0]
        total_dist, total_probs = \
            forward_batch(batch, model, ent_and_rels, ent_desc_idx, args, prediction_idx, pad_idx)
        loss = F.nll_loss(total_probs, batch.val_idx)

        if n_gpu > 1:
            loss = loss.mean()

        if not args.debug:
            meter_loss.add(loss.item(), n=batch.batch_size)

        epoch_loss += loss.item()
        correct = total_probs.max(1)[1].eq(batch.val_idx).sum()
        if not args.debug:
            meter_acc.add(correct.item() * 100, n=batch.batch_size)

        counter += batch.batch_size
        epoch_correct += correct
        if mode == 'train':
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scheduler.step()
            opt.step()

        if batch_idx % args.release_every == 0: torch.cuda.empty_cache()

        if not args.debug:

            track_txt = 'Epoch: {:3d}| Batch: {:5d}/{:5d} |'.format(epoch, batch_idx, total_batch) + \
                    ' loss: {:6.5f} |'.format(meter_loss.value()[0]) + ' acc: {:5.3f}'.format(meter_acc.value()[0])
            train_batch_logger.log(track_txt)

        else:

            track_txt = 'Epoch: {:3d}| Batch: {:5d}/{:5d} |'.format(epoch, batch_idx, total_batch) + \
                        ' loss: {:6.5f} |'.format(loss.item()) + ' acc: {:5.3f}'.format(correct.item())

            print('\r   Training | {} \r'.format(track_txt), end='\r')
    return epoch_loss, counter, epoch_correct


def train_distInfKB(train_iter, cur_epoch):
    distInfKB_train = distInfKB_model.train()
    epoch_loss, counter, epoch_correct = forward_epoch(distInfKB_train, train_iter, cur_epoch)
    return epoch_loss/counter, epoch_correct, counter


def evaluate_distInfKB(dataset_iter):
    distInfKB_eval = distInfKB_model.eval()
    epoch_loss, counter, epoch_correct = forward_epoch(distInfKB_eval, dataset_iter, mode='eval')
    return epoch_loss/counter, epoch_correct, counter


logger = logging.getLogger(__name__)

if __name__ == '__main__':
    args = process_args()
    if args.config:
        args_json = json.load(args.config)
        args.__dict__.update(dict(args_json))

    result_dir = os.path.join(args.save_path, args.name.replace('-', '_') + '_' + time.strftime('%Y_%m_%d_%H_%M'))
    checkpoint_dir = os.path.join(result_dir, "checkpoint")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(os.path.join(result_dir, 'config.json'), 'w') as configfile:
        configfile.write(json.dumps(vars(args), indent=2, sort_keys=True))

    args.id = env = 'distinfkb-' + args.name + '-' + time.strftime('%Y-%m-%d:%H:%M')

    if not args.debug:
        vsave = VisdomSaver(server=args.server, port=args.port, envs=[env])
        vtext_args = VisdomTextLogger(server=args.server, port=args.port, env=env, update_type='APPEND')
        for key, value in vars(args).items():
            vtext_args.log('{} = {}'.format(key, value))

        train_loss_logger = VisdomPlotLogger('line', server=args.server, port=args.port, env=env,
                                             opts={'title': 'Train Loss'})
        train_batch_logger = VisdomTextLogger(server=args.server, port=args.port, env=env, update_type='REPLACE',
                                              opts={'title': 'Train Batch Loss And Accuracy'})

        val_loss_logger = VisdomPlotLogger('line', server=args.server, port=args.port, env=env, opts={'title': 'Val Loss'})
        train_acc_logger = VisdomPlotLogger('line', server=args.server, port=args.port, env=env,
                                            opts={'title': 'Train Accuracy'})
        val_acc_logger = VisdomPlotLogger('line', server=args.server, port=args.port, env=env,
                                          opts={'title': 'Val Accuracy'})
        train_val_acc_logger = VisdomPlotLogger('line', server=args.server, port=args.port, env=env,
                                                opts={'title': 'Train Val Accuracy'})
        train_val_loss_logger = VisdomPlotLogger('line', server=args.server, port=args.port, env=env,
                                                 opts={'title': 'Train Val Loss'})

        meter_loss = AverageValueMeter()
        meter_acc = AverageValueMeter()

    torch.manual_seed(args.seed)
    if args.gpu > -1:
        torch.cuda.manual_seed_all(args.seed)
    print("Bert Vanilla model")

    assert args.local_rank == -1

    if args.local_rank == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
    else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')

    args.device = device

    print("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))

    ent_and_rels, distinfkb_train_iter, distinfkb_valid_iter, distinfkb_test_iter, prediction_idx, ent_desc_idx, pad_idx, \
        _, _ = init_variables(args)


    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    distInfKB_model = BertLeoModel(args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    distInfKB_model.to(args.device)

    if args.local_rank != -1:
        distInfKB_model = torch.nn.parallel.DistributedDataParallel(distInfKB_model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif n_gpu > 1:
        distInfKB_model = torch.nn.DataParallel(distInfKB_model)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    torch.cuda.empty_cache()

    print(distInfKB_model)
    print("Total number of training parameters: {}".format(sum(p.numel() for p in distInfKB_model.parameters())))
    # if args_config.gpu > -1:
    #     distInfKB_model.cuda(args_config.gpu)
    t_total = len(distinfkb_train_iter) * args.iterations
    param_optimizer = list(distInfKB_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    opt = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(opt, warmup_steps=args.warmup_steps, t_total=t_total)


    best_valid_accuracy = 0
    best_model_path = 'distinfkb.pt'
    global_start_time = time.time()
    train_loss = 0
    iter = 0
    valid_accuracy = 0
    pretrain_epoch_counter = 1
    patience = 0

    for iter in range(1, args.iterations + 1):
        if patience == args.patience:
            print("Reaching patience limit {}, stop training.".format(args.patience))
            break

        start_time = time.time()
        try:
            for train_epoch in range(1, args.train_epochs + 1):
                # train_loss, train_correct, train_count = train_distInfKB(distinfkb_train_iter)
                train_loss, train_correct, train_count = train_distInfKB(distinfkb_train_iter, train_epoch)
                valid_loss, valid_correct, valid_count = evaluate_distInfKB(distinfkb_valid_iter)
        except KeyboardInterrupt:
            import pdb;pdb.set_trace()
            option = input("What Next: -1 for exit, 0 for break, 1 for cont")
            if option == -1:
                print("Exiting")
                sys.exit()
            elif option == 0:
                print("Breaking out of DistInfKB Training")
                break
            elif option == 1:
                print("Continuing")
                continue
            else:
                print("Unknown option: " + option)
                continue

        epoch_time = time.time() - start_time

        valid_accuracy = float(valid_correct) / valid_count * 100
        print('-' * 120)
        print('| Epoch : {:3d} | globa_time: {:5.2f} | train time {:5.2f}s | train loss: {:8.7f} | train acc: {:8.5f} | valid acc: {:8.5f}' \
                .format(iter, time.time() - global_start_time, epoch_time, train_loss, float(train_correct)/train_count*100, valid_accuracy))
        print('-' * 120)
        sys.stdout.flush()

        # valid_accuracy = float(valid_correct)/valid_count
        # import pdb;pdb.set_trace()
        # Save the model if the validation loss is the best we've seen so far.
        if not best_valid_accuracy or valid_accuracy > best_valid_accuracy:
            patience = 0
            filename = get_filename('distInfKB_model',
                                    'epoch_{}_train_loss_{:.4f}_valid_acc_{:.4f}'
                                    .format(iter, train_loss, valid_accuracy))
            checkpoint(distInfKB_model, filename)
            best_model_path = checkpoint(distInfKB_model, 'distInfKB_model_best_acc.pt')
            best_valid_accuracy = valid_accuracy
        else:
            patience += 1

    vsave.save()

    filename = os.path.basename(best_model_path)
    trained_model_path = os.path.join(result_dir, filename)
    shutil.copy(best_model_path, trained_model_path)

    #########################################################################################################################################

    del distInfKB_model
    distInfKB_model = torch.load(best_model_path)

    if args.gpu > -1:
        distInfKB_model = distInfKB_model.eval()

    # Run on test data.
    start_time = time.time()
    train_loss, train_correct, train_count = evaluate_distInfKB(distinfkb_train_iter)
    train_acc = train_correct*100/train_count

    valid_loss, valid_correct, valid_count = evaluate_distInfKB(distinfkb_valid_iter)
    valid_acc = valid_correct*100/valid_count

    test_loss, test_correct, test_count = evaluate_distInfKB(distinfkb_test_iter)
    test_acc = test_correct*100.0/test_count

    test_time = time.time() - start_time

    # print test loss
    print('=' * 89)
    print('| Evaluating test set | test time: {:5.2f} | train_acc: {:5.2f}% | valid_acc: {:5.2f}% | test_acc: {:5.2f}%' \
            .format(test_time, train_acc, valid_acc, test_acc))
    print('=' * 89)
    filename = get_filename('distInfKB_model',
                            'train_acc_{:.4f}_valid_acc_{:.4f}_test_acc_{:.4f}'\
                            .format(train_acc, valid_acc, test_acc))

    checkpoint(distInfKB_model, filename)
