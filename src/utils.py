import os
import sys
import yaml
import time
import torch
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
from transformers import modeling_utils
from torch.utils.tensorboard import SummaryWriter
from alignment.data import maybe_insert_system_message, is_openai_format

NODE_RANK_COORDINATOR_PORT = 10001

_root_dir = Path(__file__).parent.parent
_conf_dir = _root_dir / 'conf'
sys.path.append(str(_conf_dir))

import settings
from settings import MODEL, LOADER, PRECISION, ACCELERATOR, OPTIMIZER

LOG = getattr(settings, 'LOG', True)
WARN = getattr(settings, 'WARN', True)
VERBOSE = getattr(settings, 'VERBOSE', False)

REMOVE_UNUSED_COLUMNS = False
ATTN_IMPL = 'flash_attention_2' if modeling_utils.is_flash_attn_2_available() else 'sdpa'

MD_FILE = 'md.pt'

ROOT_DIR = Path(__file__).parent.parent
MODEL_DIR = ROOT_DIR  / 'model'
CONF_DIR = ROOT_DIR / 'conf'
OUT_DIR = ROOT_DIR / 'outputs'
CKPT_DIR = OUT_DIR / 'checkpoints'
LOG_DIR = OUT_DIR / 'logs'
TRAIN_LOG = LOG_DIR / 'train'
TEST_LOG = LOG_DIR / 'test'
DIST_FILE = 'dist.yaml'

# Starting position for action embeddings
SUFFIX_START = 1 << 32

LOG_INTERVAL = 1

RETRY_MAX = 5

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
    model: dict
    loader: dict
    md_file: str
    log_dir: Path
    test_log: str
    train_log: str
    precision: str
    ckpt_dir: Path
    model_dir: Path
    optimizer: dict
    accelerator: str
    suffix_start: int
    log_interval: int
    remove_unused_columns: bool
        
    @property
    def lm_name(self):
        return self.model['lm']['name']
    
    @property
    def use_cache(self):
        return self.model['use_cache']
    
    @property
    def checkpoint_pretrained(self):
        return self.model['lm']['checkpoint']
    
    @property
    def skill_config(self):
        return self.model['skill'].copy()
    
    @property
    def lm_coef(self):
        return self.model.get('lm_coef', 0.7)
    
    @property
    def skill_coef(self):
        return self.model.get('skill_coef', 0.3)
    
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
    def truncation_mode(self):
        return self.loader.get('truncation_mode', 'keep_end')
    
    @property
    def po(self):
        return self.optimizer['preference']
    
    @property
    def po_conf_file(self):
        return CONF_DIR / 'simpo.yaml' if self.optimizer['preference'] == 'SimPO' else ''

cfg = Cfg(
    log=LOG,
    model=MODEL,
    loader=LOADER,
    attn=ATTN_IMPL,
    md_file=MD_FILE,
    log_dir=LOG_DIR,
    ckpt_dir=CKPT_DIR,
    test_log=TEST_LOG,
    train_log=TRAIN_LOG,
    model_dir=MODEL_DIR,
    precision=PRECISION,
    optimizer=OPTIMIZER,
    accelerator=ACCELERATOR,
    suffix_start=SUFFIX_START,
    log_interval=LOG_INTERVAL,
    remove_unused_columns=REMOVE_UNUSED_COLUMNS
)

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

def add_dist_config(
        fabric_config: dict, 
        main_addr: str = None, 
        main_port: int = None, 
        num_nodes: int = None, 
        lr: float = None, 
        weight_decay: float = None
    ):
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
        os.environ["NODE_RANK"] = str(node_rank)
        os.environ["MASTER_ADDR"] = dist_config['main_addr']
        os.environ["MASTER_PORT"] = str(dist_config['main_port'])
        fabric_config.update({'num_nodes': dist_config['num_nodes']})
    fabric_config['strategy'] = dist_config['strategy']
    if dist_config['strategy'] == 'deepspeed':
        from lightning.fabric.strategies import DeepSpeedStrategy
        if lr:
            dist_config['deepspeed']['optimizer']['params']['lr'] = lr
        if weight_decay:
            dist_config['deepspeed']['optimizer']['params']['weight_decay'] = weight_decay
        fabric_config['strategy'] = DeepSpeedStrategy(config=dist_config['deepspeed'])

def calculate_lm_loss(outputs, batch, loss_fn):
    # Get dimensions
    lm_logits = outputs['logits']
    input_ids = batch['input_ids']
    _, logits_seq_len = lm_logits.size(0), lm_logits.size(1)
    input_len = input_ids.size(1)
    
    # Calculate memory context length
    M = logits_seq_len - input_len  # Memory tokens count
    
    # Slice logits to match text sequence
    logits = lm_logits[:, M:M+input_len-1, :]  # (batch, input_len-1, vocab)
    
    # Get labels (shifted input_ids)
    labels = input_ids[:, 1:]  # (batch, input_len-1)
    
    # Create valid mask
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
        from alignment import H4ArgumentParser
        from simpo.simpo_config import SimPOConfig
        from simpo.simpo_trainer import SimPOTrainer
        parser = H4ArgumentParser((SimPOConfig,))
        training_args = parser.parse(cfg.po_conf_file)
        training_args.max_length = cfg.max_length
        training_args.max_prompt_length = cfg.max_prompt_length
        training_args.remove_unused_columns = cfg.remove_unused_columns
        return SimPOTrainer(model=model, args=training_args, tokenizer=model.tokenizer)

def md_train(model, optimizer, loader, scheduler, fabric, num_epochs=1, num_batches=-1, log_path=None, log_interval=1):
    model.train()
    po = get_po(model)
    
    metrics = {
        'total_loss': 0.0,
        'num_batches': 0
    }

    writer = SummaryWriter(log_path) if log_path else None
    lm_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    
    pbar = tqdm(
        loader,
        desc='Training',
        leave=False,
        disable=not fabric.is_global_zero,
        dynamic_ncols=True
    )

    for batch in pbar:
        optimizer.zero_grad()
        if po:
            loss, train_metrics = po.get_batch_loss_metrics(model, batch, train_eval='train', cfg=cfg)
            po.store_metrics(metrics=train_metrics, train_eval='train')
        else:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

            loss = calculate_lm_loss(outputs, batch, lm_loss_fn)
        
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                warn("NaN/Inf loss detected, skipping batch")
                continue

            has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
            has_inf = any(torch.isinf(p.grad).any() for p in model.parameters() if p.grad is not None)
            if has_nan or has_inf:
                warn("NaN/Inf gradients detected, skipping update")
                optimizer.zero_grad()
                continue
            
        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=1.0, error_if_nonfinite=False)
        optimizer.step()
        scheduler.step()
        
        metrics['total_loss'] += loss.item()
        metrics['num_batches'] += 1

        if writer:
            progress = int(metrics['num_batches'] / len(loader) * 100)
            global_step = num_epochs * len(loader) + metrics['num_batches']
            if progress % log_interval == 0 or metrics['num_batches'] == len(loader):
                writer.add_scalar("Loss/batch", loss.item(), global_step)

        if num_batches > 0:
            num_batches -= 1
            
        if num_batches == 0:
            break
    
    if po is None:
        for k in ['total_loss']:
            metrics[k] /= metrics['num_batches']
    else:
        train_metrics = po.get_metrics()['train']
        for key, val in train_metrics.items():
            metrics[key] = torch.tensor(val).mean().item()

    if writer:
        writer.close()
    
    return metrics

def md_validate(model, loader, fabric, num_batches=-1, log_path=None, log_interval=1) -> dict:
    if loader == None:
        return {}
    
    model.eval()
    po = get_po(model)

    metrics = {
        'num_batches': 0,
        'total_loss': 0.0
    }

    num_samples = 0
    writer = SummaryWriter(log_path) if log_path else None
    lm_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id, reduction='sum')

    with torch.no_grad():
        pbar = tqdm(
            loader, 
            desc='Validating', 
            disable=not fabric.is_global_zero,
            dynamic_ncols=True
        )
        for batch in pbar:
            if po:
                loss, eval_metrics = po.get_batch_loss_metrics(model, batch, train_eval='eval', cfg=cfg)
                po.store_metrics(metrics=eval_metrics, train_eval='eval')
            else:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                lm_logits = outputs['logits']
                input_len = input_ids.size(1)
                M = lm_logits.size(1) - input_len  # Memory context length
                
                sliced_lm_logits = lm_logits[:, M:M+input_len-1, :]
                lm_labels = input_ids[:, 1:]
                
                lm_mask = (attention_mask[:, 1:] != 0).flatten()
                assert sliced_lm_logits.size(1) == lm_labels.size(1), "LM dimension mismatch"

                loss = lm_loss_fn(
                    sliced_lm_logits.reshape(-1, sliced_lm_logits.size(-1))[lm_mask],
                    lm_labels.flatten()[lm_mask]
                ) if lm_mask.any() else 0.0

                num_samples += input_ids.size(0)
            
            metrics['num_batches'] += 1
            metrics['total_loss'] += loss.item()
            if writer:
                progress = int(metrics['num_batches'] / len(loader) * 100)
                if progress % log_interval == 0 or metrics['num_batches'] == len(loader):
                    writer.add_scalar("Loss/batch", loss.item(), metrics['num_batches'])
            
            if num_batches > 0:
                num_batches -= 1
            
            if num_batches == 0:
                break

    if po is None:
        gathered_loss = fabric.all_gather(torch.tensor(metrics['total_loss'])).sum()
        gathered_samples = fabric.all_gather(torch.tensor(num_samples)).sum()
        if fabric.is_global_zero and gathered_samples > 0:
            avg_loss = gathered_loss / gathered_samples
            metrics['avg_loss'] = avg_loss
            print(f"Average Loss: {avg_loss:.4f} | Samples: {gathered_samples}")

        if num_samples > 0:
            metrics['total_loss'] /= num_samples
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


def apply_chat_template(
    example,
    tokenizer,
    auto_insert_empty_system_msg: bool = True
):
    required_keys = {'chosen', 'rejected'}
    if not required_keys.issubset(example.keys()):
        raise ValueError(
            f"Could not format example as dialogue! Require either "
            f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
        )
    
    if not all(is_openai_format(msg) for msg in (example['chosen'], example['rejected'])):
        raise ValueError("Could not format example as dialogue! Require OpenAI format for all messages.")

    if 'prompt' in example and is_openai_format(example['prompt']):
        prompt_messages = example['prompt']
        chosen_messages = example['chosen']
        rejected_messages = example['rejected']
    else:
        prompt_messages = example['chosen'][:-1]
        chosen_messages = example['chosen'][-1:]
        rejected_messages = example['rejected'][-1:]

    if not (prompt_messages and chosen_messages and rejected_messages):
        raise ValueError("Prompt, chosen, and rejected must be non-empty.")
    
    if auto_insert_empty_system_msg:
        maybe_insert_system_message(prompt_messages, tokenizer)

    def apply_and_trim(messages):
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
            return text[len(tokenizer.bos_token):]
        return text
    
    example['text_prompt'] = apply_and_trim(prompt_messages)
    example['text_chosen'] = apply_and_trim(chosen_messages)
    example['text_rejected'] = apply_and_trim(rejected_messages)
    
    return example

def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device