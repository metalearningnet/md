import os
import sys
import yaml
import time
import torch
import random
import socket
import signal
import uvicorn
import requests
import subprocess
from tqdm import tqdm
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from dataclasses import dataclass
from threading import Thread, Lock
from itertools import product, islice
from typing import Dict, Optional, List
from transformers import modeling_utils
from torch.utils.tensorboard import SummaryWriter

NODE_RANK_COORDINATOR_PORT = 10001

_root_dir = Path(__file__).parent.parent
_conf_dir = _root_dir / 'conf'
sys.path.append(str(_conf_dir))

import settings
from settings import MODEL, LOADER, PRECISION, ACCELERATOR, OPTIMIZER, CKPT

LOG = getattr(settings, 'LOG', True)
WARN = getattr(settings, 'WARN', True)
VERBOSE = getattr(settings, 'VERBOSE', False)

REMOVE_UNUSED_COLUMNS = False
GRADIENT_ACCUMULATION_STEPS = 1
ATTN_IMPL = 'flash_attention_2' if modeling_utils.is_flash_attn_2_available() else 'sdpa'

MD_TAG = 'md'
MD_FILE = f'{MD_TAG}.pt'

ROOT_DIR = Path(__file__).parent.parent
MODEL_DIR = ROOT_DIR  / 'model'
CONF_DIR = ROOT_DIR / 'conf'
OUT_DIR = ROOT_DIR / 'outputs'
CKPT_DIR = OUT_DIR / 'checkpoints'
LOG_DIR = OUT_DIR / 'logs'
EVAL_DIR = OUT_DIR / 'eval'
TRAIN_LOG = LOG_DIR / 'train'
TEST_LOG = LOG_DIR / 'test'
DIST_FILE = 'dist.yaml'
PEFT_FILE = 'peft.yaml'

SEED = 42
RETRY_MAX = 5
LOG_INTERVAL = 1

LABEL_PAD_TOKEN_ID = -100
SPECIAL_TOKEN_SEP = '<|SEP|>'
RESERVED_TOKENS = [SPECIAL_TOKEN_SEP]

SYLLABLES = ['li', 'mo', 'ra', 'ba', 'ti', 'xo', 'ne', 'zu', 'ky', 'ka', 'vi', 'tho']

def info(s):
    if VERBOSE:
        print(f"[INFO] {s}")

def warn(s):
    if WARN:
        print(f"[WARN] {s}")

@dataclass
class Cfg:
    log: bool
    attn: str
    ckpt: dict
    model: dict
    loader: dict
    md_file: str
    log_dir: Path
    test_log: str
    train_log: str
    precision: str
    eval_dir: Path
    ckpt_dir: Path
    model_dir: Path
    optimizer: dict
    accelerator: str
    log_interval: int
    label_pad_token_id: int
    remove_unused_columns: bool
    gradient_accumulation_steps: int
        
    @property
    def lm_coef(self):
        return self.model.get('lm_coef', 0.7)
    
    @property
    def lm_name(self):
        return self.model['lm']['name']
    
    @property
    def lm_freeze(self):
        return self.model['lm'].get('freeze', False)
    
    @property
    def lm_checkpoint(self):
        return self.ckpt['gradient'].get('lm', False)

    @property
    def skill_config(self):
        return self.model['skill'].copy()
    
    @property
    def skill_checkpoint(self):
        return self.ckpt['gradient'].get('skill', {})
    
    @property
    def skill_integration_strategy(self):
        return self.model.get('skill_integration_strategy', 'annotation')

    @property
    def skill_coef(self):
        return self.model.get('skill_coef', 0.3)
    
    @property
    def adapter(self):
        return self.model['adapter']
    
    @property
    def temperature(self):
        return self.model.get('temperature', 0.7)

    @property
    def use_cache(self):
        return self.model['use_cache']
    
    @property
    def ckpt_path(self):
        return self.ckpt_dir / MD_FILE

    @property
    def num_reasoning_tokens(self):
        return self.model.get('num_reasoning_tokens', 5)
    
    @property
    def max_reasoning_length(self):
        return self.model.get('max_reasoning_length', 128)
    
    @property
    def min_length(self):
        return self.loader.get('min_length', 1)
    
    @property
    def max_length(self):
        return self.loader.get('max_length', 512)

    @property
    def max_prompt_length(self):
        return self.loader.get('max_prompt_length', 128)
    
    @property
    def max_target_length(self):
        return self.loader.get('max_target_length', 128)
    
    @property
    def max_annotations(self):
        return self.model.get('max_annotations', -1)
    
    @property
    def truncation_mode(self):
        return self.loader.get('truncation_mode', 'keep_end')
    
    @property
    def po(self):
        return self.optimizer.get('preference')
    
    @property
    def po_conf_file(self):
        preference = self.optimizer.get('preference')
        
        if preference == 'SimPO':
            return CONF_DIR / 'simpo.yaml'
        elif preference == 'NCA':
            return CONF_DIR / 'nca.yaml'
        else:
            return ''

cfg = Cfg(
    log=LOG,
    ckpt=CKPT,
    model=MODEL,
    loader=LOADER,
    attn=ATTN_IMPL,
    md_file=MD_FILE,
    log_dir=LOG_DIR,
    ckpt_dir=CKPT_DIR,
    test_log=TEST_LOG,
    eval_dir=EVAL_DIR,
    train_log=TRAIN_LOG,
    model_dir=MODEL_DIR,
    precision=PRECISION,
    optimizer=OPTIMIZER,
    accelerator=ACCELERATOR,
    log_interval=LOG_INTERVAL,
    label_pad_token_id=LABEL_PAD_TOKEN_ID,
    remove_unused_columns=REMOVE_UNUSED_COLUMNS,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
)

if cfg.po == 'NCA':
    from nca.nca_utils import DatasetMap
    from nca.nca_utils import nca_collate as collate
    default_dataset_path = "ChenDRAG/ultrafeedback_reward"
elif cfg.po == 'SimPO':
    from simpo.simpo_utils import DatasetMap
    from simpo.simpo_utils import simpo_collate as collate
    default_dataset_path = "princeton-nlp/gemma2-ultrafeedback-armorm"
else:
    default_dataset_path = "ag_news"

def generate_special_token_dict(min_syl=1, max_syl=3):
    def shuffled_word_list(words, seed=SEED):
        rng = random.Random(seed)
        word_list = words[:]
        rng.shuffle(word_list)
        return word_list
    seen = set()
    tokens = []
    for length in range(min_syl, max_syl + 1):
        for combo in product(SYLLABLES, repeat=length):
            token = ''.join(combo)
            if token not in seen:
                seen.add(token)
                tokens.append(token)
    return shuffled_word_list(tokens)

def get_special_token_by_index(index):
    if 0 <= index < len(SPECIAL_TOKEN_DICT):
        return f'<|{SPECIAL_TOKEN_DICT[index].capitalize()}|>'
    else:
        return None

SPECIAL_TOKEN_DICT = generate_special_token_dict()

class RegisterRequest(BaseModel):
    hostname: str

class NodeRankCoordinator:
    def __init__(self, num_nodes: int):
        self.app = FastAPI()
        self.ranks: Dict[str, int] = {}
        self.num_nodes = num_nodes
        self.lock = Lock()
        self.shutdown_flag = False
        self.setup_routes()

    def setup_routes(self):
        @self.app.post('/register')
        async def register(req: RegisterRequest):
            hostname = req.hostname
            with self.lock:
                if hostname in self.ranks:
                    return {'node_rank': self.ranks[hostname]}

                if len(self.ranks) >= self.num_nodes:
                    return {'error': 'Cluster at capacity'}, 400

                assigned_rank = len(self.ranks) + 1
                self.ranks[hostname] = assigned_rank
                info(f"Assigned rank {assigned_rank} to {hostname}")

                # Check if all ranks are assigned
                if len(self.ranks) == self.num_nodes - 1:
                    info("All ranks assigned. Shutting down node rank coordinator...")
                    Thread(target=self.graceful_shutdown, daemon=True).start()

                return {'node_rank': assigned_rank}

    def graceful_shutdown(self):
        """Give clients time to receive responses before shutting down"""
        time.sleep(2)
        os._exit(0)

    def start(self):
        free_port(NODE_RANK_COORDINATOR_PORT)
        Thread(
            target=uvicorn.run,
            args=(self.app,),
            kwargs={'host': '0.0.0.0', 'port': NODE_RANK_COORDINATOR_PORT, 'log_level': 'error'},
            daemon=True
        ).start()

def load_dist_config():
    with open(CONF_DIR / DIST_FILE) as f:
        return yaml.safe_load(f)
    
def load_peft_config():
    with open(CONF_DIR / PEFT_FILE) as f:
        return yaml.safe_load(f)

def add_dist_config(
        config: dict,
        main_addr: Optional[str] = None,
        main_port: Optional[int] = None,
        num_nodes: Optional[int] = None,
        betas: Optional[List[float]] = None,
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        lr: float = 1e-4
    ):
 
    if betas is None:
        betas = (0.9, 0.95)
    else:
        betas = tuple(float(b) for b in betas)

    dist_config = load_dist_config()

    if num_nodes != None:
        dist_config['num_nodes'] = num_nodes
    if main_port != None:
        dist_config['main_port'] = main_port
    if main_addr != None:
        dist_config['main_addr'] = main_addr
    if dist_config['num_nodes'] > 1:
        assert dist_config.get('main_addr'), (
            'Main address must be specified for distributed training or testing. '
            'Please check your configuration file.'
        )
        node_rank = get_node_rank(
            dist_config['main_addr'],
            dist_config['main_port'],
            dist_config['num_nodes']
        )
        os.environ['NODE_RANK'] = str(node_rank)
        os.environ['MASTER_ADDR'] = dist_config['main_addr']
        os.environ['MASTER_PORT'] = str(dist_config['main_port'])
        config['fabric_config'].update({'num_nodes': dist_config['num_nodes']})
    config['fabric_config']['strategy'] = dist_config['strategy']
    if dist_config['strategy'] == 'deepspeed':
        from lightning.fabric.strategies import DeepSpeedStrategy
        if 'optimizer' in dist_config['deepspeed']:
            lr = dist_config['deepspeed']['optimizer']['params'].get('lr', lr)
            eps = dist_config['deepspeed']['optimizer']['params'].get('eps', eps)
            betas = dist_config['deepspeed']['optimizer']['params'].get('betas', betas)
            weight_decay = dist_config['deepspeed']['optimizer']['params'].get('weight_decay', weight_decay)
            dist_config['deepspeed']['optimizer']['params']['lr'] = float(lr)
            dist_config['deepspeed']['optimizer']['params']['eps'] = float(eps)
            dist_config['deepspeed']['optimizer']['params']['betas'] = betas
            dist_config['deepspeed']['optimizer']['params']['weight_decay'] = float(weight_decay)
        cfg.gradient_accumulation_steps = int(dist_config.get('gradient_accumulation_steps', GRADIENT_ACCUMULATION_STEPS))
        config['fabric_config']['strategy'] = DeepSpeedStrategy(config=dist_config['deepspeed'])
        config['gradient_accumulation_steps'] = cfg.gradient_accumulation_steps

def calculate_lm_loss(outputs, batch, loss_fn):
    lm_logits = outputs['logits']
    input_ids = batch['input_ids']
    _, logits_seq_len = lm_logits.size(0), lm_logits.size(1)
    input_len = input_ids.size(1)
    
    M = logits_seq_len - input_len  # Memory tokens count
    
    logits = lm_logits[:, M:M+input_len-1, :]
    
    labels = input_ids[:, 1:]
    
    mask = (batch['attention_mask'][:, 1:] != 0).flatten()
    
    assert logits.size(1) == input_len-1, \
        f"Logits seq {logits.size(1)} != labels seq {input_len-1}"
    assert mask.shape[0] == logits.size(0)*logits.size(1), \
        f"Mask {mask.shape} vs logits {logits.shape}"
    
    return loss_fn(
        logits.reshape(-1, logits.size(-1))[mask],
        labels.reshape(-1)[mask]
    )

def get_po(model):
    if cfg.po == 'SimPO':
        from simpo.simpo_config import SimPOConfig as config
        from simpo.simpo_trainer import SimPOTrainer as trainer
    elif cfg.po == 'NCA':
        from nca.nca_config import NCAConfig as config
        from nca.nca_trainer import NCATrainer as trainer
    else:
        config = None
        trainer = None
    
    if trainer:
        from alignment import H4ArgumentParser
        parser = H4ArgumentParser((config,))
        training_args = parser.parse(cfg.po_conf_file)
        training_args.max_length = cfg.max_length
        training_args.gradient_checkpointing = False
        training_args.max_prompt_length = cfg.max_prompt_length
        training_args.remove_unused_columns = cfg.remove_unused_columns
        return trainer(
            model=model,
            args=training_args,
            tokenizer=model.tokenizer
        )

def md_train(
    model,
    loader,
    optimizer,
    fabric,
    num_epochs=1,
    num_samples=-1,
    log_path=None,
    log_interval=1,
    gradient_accumulation_steps=1
):
    model.train()
    po = get_po(model)
    
    metrics = {
        'num_batches': 0,
        'total_loss': 0.0
    }

    total_samples = 0
    writer = SummaryWriter(log_path) if log_path else None
    lm_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    last_log_step = 0

    for epoch in range(num_epochs):
        epoch_samples = 0

        if num_samples > 0:
            max_batches = num_samples
            loader_iter = iter(islice(loader, max_batches))
            pbar_total = max_batches
        else:
            loader_iter = iter(loader)
            pbar_total = len(loader)

        pbar = tqdm(
            loader_iter,
            desc=f'Training Epoch {epoch + 1}/{num_epochs}',
            disable=not fabric.is_global_zero,
            dynamic_ncols=True,
            total=pbar_total
        )

        for step, batch in enumerate(pbar):
            if po:
                loss, train_metrics = po.get_batch_loss_metrics(model, batch, train_eval='train')
                po.store_metrics(metrics=train_metrics, train_eval='train')
            else:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = calculate_lm_loss(outputs, batch, lm_loss_fn)

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    warn("NaN/Inf loss detected, skipping batch")
                    continue

                has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
                has_inf = any(torch.isinf(p.grad).any() for p in model.parameters() if p.grad is not None)
                if has_nan or has_inf:
                    warn("NaN/Inf gradients detected, skipping update")
                    continue

            fabric.backward(loss)
            if optimizer:
                optimizer.step()
                optimizer.zero_grad()
            else:
                if (step + 1) % gradient_accumulation_steps == 0:
                    model._forward_module.step()

            total_samples += 1
            epoch_samples += 1

            metrics['total_loss'] += loss.item()
            metrics['num_batches'] += 1

            if writer:
                if num_samples > 0:
                    sample_progress = int((total_samples / (num_samples * num_epochs)) * 100)
                else:
                    estimated_total = len(loader) * num_epochs
                    sample_progress = int((total_samples / estimated_total) * 100)

                if sample_progress >= log_interval and (sample_progress // log_interval) > (last_log_step // log_interval):
                    writer.add_scalar("Loss/batch", loss.item(), metrics['num_batches'])
                    last_log_step = sample_progress
        
        pbar.close()

    if po is None:
        if metrics['num_batches'] > 0:
            metrics['total_loss'] /= metrics['num_batches']
    else:
        train_metrics = po.get_metrics()['train']
        for key, val in train_metrics.items():
            metrics[key] = torch.tensor(val).mean().item()

    if writer:
        writer.close()

    return metrics

def md_validate(
    model, 
    loader,
    fabric,
    num_samples=-1,
    log_path=None,
    log_interval=1
) -> dict:
    if loader == None:
        return {}
    
    model.eval()
    po = get_po(model)

    metrics = {
        'num_batches': 0,
        'total_loss': 0.0
    }

    total_samples = 0
    writer = SummaryWriter(log_path) if log_path else None
    lm_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id, reduction='sum')

    with torch.no_grad():
        if num_samples > 0:
            max_batches = num_samples
            loader_iter = iter(islice(loader, max_batches))
            pbar_total = max_batches
        else:
            loader_iter = iter(loader)
            pbar_total = len(loader)
        
        pbar = tqdm(
            loader_iter,
            desc='Validating',
            disable=not fabric.is_global_zero,
            dynamic_ncols=True,
            total=pbar_total
        )
        
        for batch in pbar:
            if po:
                loss, eval_metrics = po.get_batch_loss_metrics(model, batch, train_eval='eval')
                po.store_metrics(metrics=eval_metrics, train_eval='eval')
            else:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                lm_logits = outputs['logits']
                input_len = input_ids.size(1)
                M = lm_logits.size(1) - input_len
                
                sliced_lm_logits = lm_logits[:, M:M+input_len-1, :]
                lm_labels = input_ids[:, 1:]
                
                lm_mask = (attention_mask[:, 1:] != 0).flatten()
                assert sliced_lm_logits.size(1) == lm_labels.size(1), "LM dimension mismatch"

                loss = lm_loss_fn(
                    sliced_lm_logits.reshape(-1, sliced_lm_logits.size(-1))[lm_mask],
                    lm_labels.flatten()[lm_mask]
                ) if lm_mask.any() else 0.0
            
            total_samples += 1
            metrics['num_batches'] += 1
            metrics['total_loss'] += loss.item()

            if writer:
                if num_samples > 0:
                    sample_progress = int((total_samples / num_samples) * 100)
                else:
                    sample_progress = int((total_samples / len(loader)) * 100)

                if sample_progress % log_interval == 0:
                    writer.add_scalar("Loss/batch", loss.item(), metrics['num_batches'])

    if po is None:
        gathered_loss = fabric.all_gather(torch.tensor(metrics['total_loss'])).sum()
        gathered_samples = fabric.all_gather(torch.tensor(total_samples)).sum()
        if fabric.is_global_zero and gathered_samples > 0:
            avg_loss = gathered_loss / gathered_samples
            metrics['avg_loss'] = avg_loss
            print(f"Average Loss: {avg_loss:.4f} | Samples: {gathered_samples}")

        if metrics['num_batches'] > 0:
            metrics['total_loss'] /= metrics['num_batches']
    else:
        eval_metrics = po.get_metrics()['eval']
        for key, val in eval_metrics.items():
            metrics[key] = torch.tensor(val).mean().item()
            gathered_vals = fabric.all_gather(torch.tensor(val))
            if fabric.is_global_zero:
                metrics[f'avg_{key}'] = gathered_vals.mean().item()
                print(f"avg_{key}: {metrics[f'avg_{key}']}")

    if writer:
        writer.close()
    
    return metrics

def free_port(port):
    try:
        result = subprocess.check_output(
            ["lsof", "-t", f"-i:{port}"], text=True
        ).strip()
        
        if result:
            pids = result.split("\n")
            info(f"Port {port} is in use by PID(s): {pids}")
            for pid in pids:
                os.kill(int(pid), signal.SIGKILL)
            info(f"Port {port} is now freed.")
    except subprocess.CalledProcessError:
        pass

def get_node_rank(main_address: str, main_port: int, num_nodes: int) -> int:
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    is_main_node = (host_ip == socket.gethostbyname(main_address))

    if is_main_node:
        info("Starting node rank coordinator...")
        free_port(main_port)
        coordinator = NodeRankCoordinator(num_nodes)
        coordinator.start()
        return 0
    else:
        for attempt in range(RETRY_MAX):
            try:
                response = requests.post(
                    f"http://{main_address}:{NODE_RANK_COORDINATOR_PORT}/register",
                    json={'hostname': hostname},
                    timeout=2
                )
                if response.status_code == 200:
                    info(f'node_rank: {response.json()['node_rank']}')
                    return response.json()['node_rank']
            except (requests.ConnectionError, requests.Timeout):
                if attempt == RETRY_MAX - 1:
                    break
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to contact node rank coordinator at {main_address}")

def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device
