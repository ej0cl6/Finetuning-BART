import os, sys, json, logging, datetime, pprint, _jsonnet, subprocess, random
import numpy as np
import torch
import rouge
from torch.utils.data import DataLoader
from transformers import BartTokenizer, AdamW, get_linear_schedule_with_warmup
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from model import FinetuningBART
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', required=True)
parser.add_argument('--seed', type=int, required=False)
args = parser.parse_args()

config = json.loads(_jsonnet.evaluate_file(args.config))
config = Namespace(**config)

if args.seed is not None:
    config.seed = args.seed

# fix random seed
random.seed(0)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

# set GPU device
torch.cuda.set_device(config.gpu_device)

# logger and summarizer
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]
output_dir = os.path.join(config.output_dir, timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_path = os.path.join(output_dir, "train.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]', 
                    handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")

# output
with open(os.path.join(output_dir, "config.json"), 'w') as fp:
    json.dump(vars(config), fp, indent=4)
    
def load_data(src_file, tgt_file, config):
    with open(src_file) as fp:
        src_lines = fp.readlines()
    src_lines = [l.strip() for l in src_lines]
    with open(tgt_file) as fp:
        tgt_lines = fp.readlines()
    tgt_lines = [l.strip() for l in tgt_lines]
    assert len(src_lines) == len(tgt_lines)
    n_data = len(src_lines)
    
    new_src_lines, new_tgt_lines = [], []
    tokenizer = BartTokenizer.from_pretrained(config.pretrained_model, cache_dir=config.cache_dir)
    for src_line, tgt_line in tqdm(zip(src_lines, tgt_lines), total=n_data, ncols=100):
        src_len = len(tokenizer(src_line)["input_ids"])
        tgt_len = len(tokenizer(tgt_line)["input_ids"])
        
        if src_len > config.max_src_len or tgt_len > config.max_tgt_len:
            continue
            
        new_src_lines.append(src_line)
        new_tgt_lines.append(tgt_line)
    
    n_new_data = len(new_src_lines)
    logger.info(f"Load {n_new_data}/{n_data} instances from {src_file} and {tgt_file}")

    
    return src_lines, tgt_lines, n_data
    
def evaluate(epoch, model, eval_data, config, mode, show=True):
    model.eval()
    avg_loss = 0.0
    eval_outputs = []
    with torch.no_grad():
        eval_loader = DataLoader(np.arange(eval_data[2]), batch_size=config.eval_batch_size, shuffle=False)
        for bid, eval_idxs in tqdm(enumerate(eval_loader), total=len(eval_loader), ncols=100, ascii=True):

            src_sents = [eval_data[0][i] for i in eval_idxs]
            tgt_sents = [eval_data[1][i] for i in eval_idxs]
            
            loss = model(src_sents, tgt_sents)
            avg_loss += loss.item()
            outputs = model.generate(src_sents)
            eval_outputs.extend(outputs)
            
    avg_loss /= len(eval_loader)
            
    eval_scores = {"loss": avg_loss}
    
    if show:
        print("-------------------------------------------------------")
        print(f"Epoch {epoch} {mode.capitalize()}")
        print("-------------------------------------------------------")
        print("LOSS:   {:6.3f}".format(eval_scores['loss']))
        print("-------------------------------------------------------")
    
    return eval_scores, eval_outputs
            
train_data = load_data(config.train_src_file, config.train_tgt_file, config)
dev_data = load_data(config.dev_src_file, config.dev_tgt_file, config)
test_data = load_data(config.test_src_file, config.test_tgt_file, config)

model = FinetuningBART(config)
model.cuda(device=config.gpu_device)

# optimizer
param_groups = [{'params': model.parameters(), 'lr': config.learning_rate, 'weight_decay': config.weight_decay}]
optimizer = AdamW(params=param_groups)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=(len(train_data)//config.train_batch_size+(len(train_data)%config.train_batch_size!=0)) * config.warmup_epoch,
                                            num_training_steps=(len(train_data)//config.train_batch_size+(len(train_data)%config.train_batch_size!=0)) * config.max_epoch)


# start training
logger.info("Start training ...")
best_dev_scores = {"loss": np.inf}
best_test_scores = {"loss": np.inf}
best_dev_epoch = -1
for epoch in range(1, config.max_epoch+1):
    logger.info(log_path)
    logger.info(f"Epoch {epoch}")
    
    train_loader = DataLoader(np.arange(train_data[2]), batch_size=config.train_batch_size, shuffle=True)
    
    model.train()
    
    avg_loss = 0.0
    
    for bid, train_idxs in tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, ascii=True):

        src_sents = [train_data[0][i] for i in train_idxs]
        tgt_sents = [train_data[1][i] for i in train_idxs]
        
        # forard model
        loss = model(src_sents, tgt_sents)
        avg_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clipping)
        optimizer.step()
        scheduler.step()
        
    avg_loss /= len(train_loader)
    print(f"Loss: {avg_loss}")
        
    
    # eval dev set
    model.eval()
    dev_scores, dev_outputs = evaluate(epoch, model, dev_data, config, "dev")
    logger.info({"epoch": epoch, "dev_scores": dev_scores})
    
    if dev_scores["loss"] < best_dev_scores["loss"]:
        logger.info(f"Saving best model to {output_dir}")
        model.save(output_dir)
        
        # eval test set
        test_scores, test_outputs = evaluate(epoch, model, test_data, config, "test")
        logger.info({"epoch": epoch, "test_scores": test_scores})
        
        best_dev_scores = dev_scores
        best_test_scores = test_scores
        best_dev_epoch = epoch
        
        with open(os.path.join(output_dir, "dev_predictions.txt"), "w") as fp:
            for dev_output in dev_outputs:
                fp.write(dev_output+"\n")
        
        with open(os.path.join(output_dir, "test_predictions.txt"), "w") as fp:
            for test_output in test_outputs:
                fp.write(test_output+"\n")
    
    logger.info("Current best")
    logger.info({"best_epoch": best_dev_epoch, "best_dev_scores": best_dev_scores, "best_test_scores": best_test_scores})
        
logger.info(log_path)
logger.info("Done!")
