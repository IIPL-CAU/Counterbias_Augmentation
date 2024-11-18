# Import modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import logging
import numpy as np
from tqdm import tqdm
from time import time
# Import PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# Import custom modules
from model.dataset import CustomSeq2seqDataset
from optimizer.utils import shceduler_select, optimizer_select
from utils import TqdmLoggingHandler, write_log, get_tb_exp_name, return_model_name
from task.utils import data_load, data_sampling, aug_data_load, tokenizing, input_to_device, encoder_parameter_grad, result_writing

def tst_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, 'Start training!')

    #===================================#
    #=============Data Load=============#
    #===================================#

    write_log(logger, "Load data...")

    start_time = time()
    total_src_list, total_trg_list = data_load(args)
    num_labels = len(set(total_trg_list['train']))
    if args.train_with_aug:
        aug_src_list, aug_label_list = aug_data_load(args)
        total_src_list['train'] = np.concatenate((total_src_list['train'], np.array(aug_src_list['train'])))
        total_trg_list['train'] = np.concatenate((total_trg_list['train'], np.array(aug_label_list)))

    write_log(logger, 'Data loading done!')

    #===================================#
    #===========Train setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, 'Instantiating model...')
    model_name = 'google-t5/t5-base'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    # 2) Dataloader setting
    tokenizer_name = 'google-t5/t5-base'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dataset_dict = {
        'train': CustomSeq2seqDataset(tokenizer=tokenizer,
                               src_list=total_src_list['train'], trg_list=total_trg_list['train'], 
                               src_max_len=args.src_max_len, trg_max_len=args.trg_max_len),
        'valid': CustomSeq2seqDataset(tokenizer=tokenizer,
                               src_list=total_src_list['valid'], trg_list=total_trg_list['valid'], 
                               src_max_len=args.src_max_len, trg_max_len=args.trg_max_len),
        'test': CustomSeq2seqDataset(tokenizer=tokenizer,
                               src_list=total_src_list['test'], trg_list=total_trg_list['test'], 
                               src_max_len=args.src_max_len, trg_max_len=args.trg_max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.batch_size, shuffle=True,
                            pin_memory=True, num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=True, 
                            pin_memory=True, num_workers=args.num_workers),
        'test': DataLoader(dataset_dict['test'], drop_last=False,
                           batch_size=args.batch_size, shuffle=False, 
                           pin_memory=True, num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    # 3) Optimizer & Learning rate scheduler setting
    optimizer = optimizer_select(optimizer_model=args.cls_optimizer, model=model, lr=args.cls_lr, w_decay=args.w_decay)
    scheduler = shceduler_select(phase='training', scheduler_model=args.cls_scheduler, optimizer=optimizer, dataloader_len=len(dataloader_dict['train']), args=args)

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss(label_smoothing=args.cls_label_smoothing_eps).to(device)

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        write_log(logger, 'Resume model...')
        save_path = os.path.join(args.model_save_path, args.data_name)
        save_file_name = os.path.join(save_path,
                                        f'checkpoint_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}_v_{args.variational_mode}_p_{args.parallel}.pth.tar')
        checkpoint = torch.load(save_file_name)
        start_epoch = checkpoint['epoch'] - 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    write_log(logger, 'Training start!')
    best_val_loss = 1e+4
    best_val_acc = 0

    for epoch in range(start_epoch + 1, args.training_num_epochs + 1):
        start_time_e = time()

        write_log(logger, 'Training start...')
        model.train()

        for i, batch_iter in enumerate(tqdm(dataloader_dict['train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

            optimizer.zero_grad(set_to_none=True)

            # Input setting
            src_sequence = batch_iter[0].to(device)
            src_att = batch_iter[1].to(device)
            trg_label = batch_iter[2].to(device)

            #===================================#
            #=======Classifier Training=========#
            #===================================#

            # Augmenter training
            trg_label = model._shift_right(trg_label)
            out = model(input_ids=src_sequence, attention_mask=src_att, labels=trg_label)

            logit = out['logits']
            loss = out['loss']
            loss.backward()
            if args.clip_grad_norm > 0:
                clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            scheduler.step()

            # Accuracy
            acc = sum(logit.argmax(dim=2).flatten() == trg_label.flatten()) / len(trg_label.flatten())

            # Print loss value only training
            if i == 0 or i % args.print_freq == 0 or i == len(dataloader_dict['train'])-1:
                iter_log = "[Epoch:%03d][%03d/%03d] train_loss:%03.2f | train_acc:%03.2f | learning_rate:%1.6f | spend_time:%02.2fmin" % \
                    (epoch, i, len(dataloader_dict['train'])-1, loss.item(), acc.item(), optimizer.param_groups[0]['lr'], (time() - start_time_e) / 60)
                write_log(logger, iter_log)

        #===================================#
        #============Validation=============#
        #===================================#
        write_log(logger, 'Validation start...')

        # Validation
        model.eval()
        val_acc = 0
        val_loss = 0
        for batch_iter in tqdm(dataloader_dict['valid'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):

            # Input setting
            src_sequence = batch_iter[0].to(device)
            src_att = batch_iter[1].to(device)
            trg_label = batch_iter[2].to(device)

            with torch.no_grad():
                # Classifier Training
                trg_label = model._shift_right(trg_label)
                out = model(input_ids=src_sequence, attention_mask=src_att, labels=trg_label)

                logit = out['logits']
                loss = out['loss']

                val_loss += loss
                val_acc += sum(logit.argmax(dim=2).flatten() == trg_label.flatten()) / len(trg_label.flatten())

        val_loss /= len(dataloader_dict['valid'])
        val_acc /= len(dataloader_dict['valid'])
        write_log(logger, 'Augmenter Validation CrossEntropy Loss: %3.3f' % val_loss)
        write_log(logger, 'Augmenter Validation Accuracy: %3.3f' % val_acc)

        save_file_name = os.path.join(f'tst_training_checkpoint_seed_{args.random_seed}_aug_{args.train_with_aug}_{args.aug_method}.pth.tar')
        if val_loss < best_val_loss:
            write_log(logger, 'Checkpoint saving...')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, save_file_name)
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
        else:
            else_log = f'Still {best_epoch} epoch Loss({round(best_val_loss.item(), 2)}) is better...'
            write_log(logger, else_log)

    #===================================#
    #=========Model Test Start==========#
    #===================================#
    write_log(logger, 'Test!')

    # Generate predictions for the test set
    model.eval()
    prediction_list = []
    test_acc = 0
    
    for batch_iter in tqdm(dataloader_dict['test'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):
        # Input setting
        src_sequence = batch_iter[0].to(device)
        src_att = batch_iter[1].to(device)
        trg_label = batch_iter[2].to(device)

        with torch.no_grad():
            #  convert the model output logits to predicted class labels
            trg_label = model._shift_right(trg_label)
            out = model(input_ids=src_sequence, attention_mask=src_att, labels=trg_label)

            logit = out['logits']
            loss = out['loss']

            predictions = tokenizer.batch_decode(logit.argmax(dim=2), skip_special_tokens=True)
            prediction_list.extend(predictions.detach().cpu().tolist())

            test_acc += sum(logit.argmax(dim=2).flatten() == trg_label.flatten()) / len(trg_label.flatten())
            
            # Convert predictions to a list of strings
            # for prediction in predictions:
            #     if prediction == 0:
            #         prediction_strings.append('entailment')
            #     else:
            #         prediction_strings.append('not_entailment')

    test_acc /= len(dataloader_dict['test'])
    write_log(logger, 'Test Accuracy: %3.3f' % test_acc)
    
    # Write predictions to a tsv file with index, prediction string header
    save_file_name = os.path.join(args.result_path, args.data_name, f'{args.random_seed}.tsv')
    with open(save_file_name, 'wt') as f:
        f.write('index\tprediction\n')
        for idx, prediction in enumerate(prediction_list):
            f.write(f'{idx}\t{prediction}\n')

    write_log(logger, f'Test predictions saved --> {save_file_name}')        

    # 3) Results
    write_log(logger, f'Best Epoch: {best_epoch}')
    write_log(logger, f'Best Loss: {round(best_val_loss.item(), 2)}')
    write_log(logger, f'Best acc: {round(best_val_acc.item(), 2)}')

    task = 'cls'
    result_writing(args, task, best_val_acc, best_val_loss)