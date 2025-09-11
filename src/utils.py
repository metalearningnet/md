import os
import sys
import yaml
import time
import copy
import torch
import shutil
import random
import socket
import signal
import uvicorn
import requests
import subprocess
import transformers
from tqdm import tqdm
from pathlib import Path
from fastapi import FastAPI
from packaging import version
from pydantic import BaseModel
import torch.nn.functional as F
from dataclasses import dataclass
from threading import Thread, Lock
from itertools import product, islice
from typing import Dict, Optional, List, Union
from torch.utils.tensorboard import SummaryWriter
from lightning.fabric.accelerators import CUDAAccelerator
from transformers import (
    modeling_utils,
    TopKLogitsWarper,
    TopPLogitsWarper,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    RepetitionPenaltyLogitsProcessor
)

NODE_RANK_COORDINATOR_PORT = 10001

_root_dir = Path(__file__).parent.parent
_conf_dir = _root_dir / 'conf'
sys.path.append(str(_conf_dir))

import settings
from chat import VOCAB, TEMPLATE
from settings import MODEL, LOADER, PRECISION, OPTIMIZER, CKPT, MEMORY

LOG = getattr(settings, 'LOG', False)
WARN = getattr(settings, 'WARN', True)
VERBOSE = getattr(settings, 'VERBOSE', True)

SHOW_INPUT = False
SHOW_OUTPUT = False
SHOW_LABELS = False

REMOVE_UNUSED_COLUMNS = False
GRADIENT_ACCUMULATION_STEPS = 1

MD_TAG = 'md'
MD_FILE = f'{MD_TAG}.pt'

ROOT_DIR = Path(__file__).parent.parent
MODEL_DIR = ROOT_DIR / 'models'
CONF_DIR = ROOT_DIR / 'conf'
OUT_DIR = ROOT_DIR / 'output'
LOG_DIR = OUT_DIR / 'logs'

EVAL_DIR = OUT_DIR / 'results'
CKPT_DIR = OUT_DIR / 'checkpoints'

TEST_LOG = LOG_DIR / 'test'
TRAIN_LOG = LOG_DIR / 'train'

LM_DIR = MODEL_DIR / 'lm'

MAC_FILE = 'mac.yaml'
MAL_FILE = 'mal.yaml'
DIST_FILE = 'dist.yaml'
PEFT_FILE = 'peft.yaml'

SEED = 42
RETRY_MAX = 5
LOG_INTERVAL = 1

SEP_TOKEN = '<separator>'
RESERVED_TOKENS = [SEP_TOKEN]
LABEL_PAD_TOKEN_ID = -100

SYLLABLES = [
    'jho', 'tsu', 'vlu', 'bya',
    'stu', 'vya', 'plo', 'ske', 'blu', 'twy',
    'ghy', 'klo', 'zhi', 'fli', 'spu', 'dwe',
    'xo', 'zu', 'ky', 'vle', 'nz', 'mra', 'xy',
    'fyo', 'twu', 'pyo', 'nyo', 'kwe', 'psu', 'vri'
]

BOUNDARY_TOKENS = ['\n']

BOLD = '\033[1m'
RESET = '\033[0m'
BLUE = '\033[1;34m'
YELLOW = '\033[1;33m'

STRATEGY = 'deepspeed' # Options: 'deepspeed' | 'ddp'

with open(_conf_dir / MAC_FILE) as f:
    default_mac_config = yaml.safe_load(f)

with open(_conf_dir / MAL_FILE) as f:
    default_mal_config = yaml.safe_load(f)

def info(s):
    if VERBOSE:
        print(f"{BLUE}{BOLD}[INFO]{RESET} {s}")

def warn(s):
    if WARN:
        print(f"{YELLOW}{BOLD}[WARNING]{RESET} {s}", file=sys.stderr)

def show_input(s):
    if VERBOSE:
        print(f'> [input] {s}')

def show_labels(s):
    if VERBOSE:
        print(f'\n[lables] {s}\n')

def show_output(s):
    if VERBOSE:
        print(f'< [output] {s}')

@dataclass
class Cfg:
    log: bool
    ckpt: dict
    model: dict
    loader: dict
    memory: dict
    md_file: str
    lm_dir: Path
    log_dir: Path
    test_log: str
    train_log: str
    precision: str
    eval_dir: Path
    ckpt_dir: Path
    optimizer: dict
    log_interval: int
    remove_unused_columns: bool
    
    def mem_config(self, component, mem_type):
        valid_components = ['frontend', 'backend']
        config_mapping = {
            'mac': (default_mac_config, 'mac'),
            'mal': (default_mal_config, 'mal'),
        }

        if mem_type not in config_mapping:
            valid_types = ', '.join(config_mapping.keys())
            raise ValueError(f"Invalid memory type: '{mem_type}'. Valid types are: {valid_types}")

        if component not in valid_components:
            raise ValueError(f"Invalid component: '{component}'. Valid components are: {valid_components}")

        default_config, config_key = config_mapping[mem_type]
        config = copy.deepcopy(default_config)

        custom_config = self.memory[component]['memory'][config_key]
        config.update(custom_config)

        return config
    
    @property
    def attn(self):
        if self.lm_name.startswith('gemma'):
            return 'eager'
        elif modeling_utils.is_flash_attn_2_available():
            return 'flash_attention_2'
        else:
            return 'sdpa'
    
    @property
    def lm_checkpoint(self):
        return self.ckpt['gradient'].get('lm', False)
    
    @property
    def lm_path(self):
        return self.model['lm']['path']
    
    @property
    def lm_name(self):
        return os.path.basename(self.lm_path)
    
    @property
    def lm_coef(self):
        return self.model.get('lm_coef', 0.8)
    
    @property
    def lm_freeze(self):
        return self.model['lm'].get('freeze', False)
    
    @property
    def lm_temperature(self):
        return self.model['lm'].get('temperature')
    
    @property
    def min_length(self):
        return self.model['lm'].get('min_length', 64)
    
    @property
    def max_length(self):
        return self.model['lm'].get('max_length', 1280)

    @property
    def max_prompt_length(self):
        return self.model['lm'].get('max_prompt_length', 512)
    
    @property
    def max_target_length(self):
        return self.model['lm'].get('max_target_length', 1024)
    
    @property
    def mem_coef(self):
        return self.model.get('mem_coef', 0.2)
    
    @property
    def strategy_info(self):
        backend_strategy = self.backend_strategy
        if backend_strategy == 'hint':
            backend_strategy = f'{backend_strategy}-{self.backend_hint_category}'
        return backend_strategy
    
    @property
    def has_frontend(self):
        return self.model['frontend']

    @property
    def use_cache(self):
        return self.model['use_cache']
    
    @property
    def use_initial_prompt(self):
        return self.model['use_initial_prompt']
    
    @property
    def sft(self):
        return self.model.get('sft', False)
    
    @property
    def ckpt_path(self):
        return self.ckpt_dir / MD_FILE
    
    @property
    def frontend_mem_type(self):
        return self.memory['frontend']['memory']['type']
    
    @property
    def frontend_mem_config(self):
        return self.mem_config('frontend', self.frontend_mem_type)
    
    @property
    def frontend_update_memory(self):
        return self.memory['frontend']['memory'].get('update', True)
    
    @property
    def backend_vocab(self):
        if self.backend_strategy == 'hint':
            return VOCAB['hint']
    
    @property
    def backend_skill_config(self):
        return self.memory['backend']['skill']
    
    @property
    def backend_mem_type(self):
        return self.memory['backend']['memory']['type']
    
    @property
    def backend_mem_config(self):
        return self.mem_config('backend', self.backend_mem_type)
    
    @property
    def backend_fusion_adapter(self):
        return self.memory['backend']['strategy']['fusion']['adapter']
    
    @property
    def backend_anno_max_length(self):
        return self.memory['backend']['strategy']['annotation'].get('max_length', 3)
    
    @property
    def backend_max_annotations(self):
        return self.memory['backend']['strategy']['annotation'].get('max_annotations', -1)
    
    @property
    def backend_strategy(self):
        return self.memory['backend']['strategy'].get('type', 'hint')
    
    @property
    def backend_hint_category(self):
        return self.memory['backend']['strategy']['hint']['category']
    
    @property
    def backend_max_hints(self):
        return self.memory['backend']['strategy']['hint'].get('max_hints', -1)
    
    @property
    def backend_special_tokens(self):
        if self.backend_strategy == 'annotation':
            return self.memory['backend']['strategy']['annotation'].get('words', 8)
        elif self.backend_strategy == 'hint':
            return len(VOCAB['hint'][self.backend_hint_category])
        else:
            return 0
    
    @property
    def backend_context_window(self):
        return self.memory['backend'].get('context_window', 4)
    
    @property
    def backend_update_memory(self):
        return self.memory['backend']['memory'].get('update', True)
    
    @property
    def backend_tune_special_token_embeddings(self):
        if self.backend_strategy == 'hint':
            return self.memory['backend']['strategy']['hint'].get('tune', False)
        elif self.backend_strategy == 'annotation':
            return self.memory['backend']['strategy']['annotation'].get('tune', False)
        else:
            return False
        
    @property
    def backend_sentence_alignment(self):
        if self.backend_strategy == 'hint':
            return self.memory['backend']['strategy']['hint'].get('sentence_alignment')
    
    @property
    def backend_min_interval(self):
        if self.backend_strategy == 'hint':
            return self.memory['backend']['strategy']['hint'].get('min_interval', 8)
        elif self.backend_strategy == 'annotation':
            return self.memory['backend']['strategy']['annotation'].get('min_interval', 8)
        else:
            return 1
    
    @property
    def backend_sep_logit_bias(self):
        if self.backend_strategy == 'hint':
            return self.memory['backend']['strategy']['hint'].get('sep_logit_bias', 0.0)
        else:
            return 0.0
    
    @property
    def backend_sep_temperature(self):
        if self.backend_strategy == 'hint':
            return self.memory['backend']['strategy']['hint'].get('sep_temperature', 1.0)
        else:
            return 1.0
    
    @property
    def backend_checkpoint(self):
        return self.ckpt['gradient']['mem']
    
    @property
    def truncation_mode(self):
        return self.loader.get('truncation_mode', 'keep_end')
    
    @property
    def po(self):
        return self.optimizer.get('preference')
    
    @property
    def lr(self):
        return self.optimizer['gradient'].get('lr', 3e-5)
    
    @property
    def eps(self):
        return self.optimizer['gradient'].get('eps', 1e-6)
    
    @property
    def betas(self):
        return self.optimizer['gradient'].get('betas', (0.9, 0.95))
    
    @property
    def weight_decay(self):
        return self.optimizer['gradient'].get('weight_decay', 0.1)
    
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
    memory=MEMORY,
    loader=LOADER,
    lm_dir=LM_DIR,
    md_file=MD_FILE,
    log_dir=LOG_DIR,
    ckpt_dir=CKPT_DIR,
    test_log=TEST_LOG,
    eval_dir=EVAL_DIR,
    train_log=TRAIN_LOG,
    precision=PRECISION,
    optimizer=OPTIMIZER,
    log_interval=LOG_INTERVAL,
    remove_unused_columns=REMOVE_UNUSED_COLUMNS
)

if not cfg.sft and cfg.po == 'NCA':
    from nca.nca_utils import DatasetMap
    from nca.nca_utils import nca_collate as collate
    default_dataset_path = "ChenDRAG/ultrafeedback_reward"
elif not cfg.sft and cfg.po == 'SimPO':
    from simpo.simpo_utils import DatasetMap
    from simpo.simpo_utils import simpo_collate as collate
    default_dataset_path = "metalearningnet/qwen3-ultrafeedback"
else:
    default_dataset_path = "databricks/databricks-dolly-15k"
    from transformers import PreTrainedTokenizer

    @dataclass
    class DatasetMap:
        tokenizer: PreTrainedTokenizer
        truncation_mode: str
        max_prompt_length: int
        max_length: int
        max_target_length: int
        label_pad_token_id: int = -100
        
        def __call__(self, examples: dict) -> dict:
            input_ids = []
            labels = []

            if 'instruction' in examples and 'response' in examples:
                prompts = []
                responses = []
                
                for instruction, response in zip(examples['instruction'], examples['response']):
                    prompt = f"Instruction: {instruction}\n\nResponse:"
                    prompts.append(prompt)
                    responses.append(response)

                prompt_encodings = self.tokenizer(
                    prompts,
                    truncation=True,
                    max_length=self.max_prompt_length,
                    padding=False,
                    add_special_tokens=False
                )

                response_encodings = self.tokenizer(
                    responses,
                    truncation=True,
                    max_length=self.max_target_length,
                    padding=False,
                    add_special_tokens=False
                )

                input_ids = []
                labels = []
                for prompt_ids, response_ids in zip(prompt_encodings['input_ids'], response_encodings['input_ids']):
                    combined_ids = prompt_ids + response_ids
                    
                    if len(combined_ids) > self.max_length:
                        if self.truncation_mode == 'keep_start':
                            combined_ids = combined_ids[:self.max_length]
                        else:
                            combined_ids = combined_ids[-self.max_length:]
                    
                    # Create labels (-100 for prompt, response tokens for response)
                    label_ids = [self.label_pad_token_id] * len(prompt_ids) + response_ids
                    label_ids = label_ids[:len(combined_ids)]
                    
                    input_ids.append(combined_ids)
                    labels.append(label_ids)
            elif 'text' in examples:
                texts = examples['text']
                text_encodings = self.tokenizer(
                    texts,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                    add_special_tokens=False,
                )

                for text_ids in text_encodings["input_ids"]:
                    if len(text_ids) > self.max_length:
                        if self.truncation_mode == "keep_start":
                            text_ids = text_ids[: self.max_length]
                        else:
                            text_ids = text_ids[-self.max_length :]
                    
                    input_ids.append(text_ids)
                    labels.append(text_ids.copy())
            else:
                raise ValueError("Examples must contain either ('instruction' + 'response') or 'text'.")
            
            return {
                'input_ids': input_ids,
                'attention_mask': [[1] * len(ids) for ids in input_ids],
                'labels': labels
            }

        def generate(self, dataset):
            return dataset.map(
                self.__call__,
                batched=True,
                remove_columns=dataset.column_names,
                num_proc=os.cpu_count()
            )

    def collate(
        batch: List[Dict[str, Union[List[int], torch.Tensor]]],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        label_pad_token_id: int = -100,
        truncation_mode: str = 'keep_start',
        is_encoder_decoder: bool = False
    ) -> Dict[str, torch.Tensor]:
        input_ids = [torch.as_tensor(item['input_ids']) for item in batch]
        labels = [torch.as_tensor(item['labels']) for item in batch]
        
        seq_lengths = torch.tensor([len(seq) for seq in input_ids])
        max_len = seq_lengths.max().item()
        
        if max_length is not None:
            max_len = min(max_len, max_length)
            seq_lengths = torch.clamp(seq_lengths, max=max_len)
        
        input_ids_padded = torch.full(
            (len(batch), max_len),
            fill_value=tokenizer.pad_token_id,
            dtype=input_ids[0].dtype
        )
        labels_padded = torch.full(
            (len(batch), max_len),
            fill_value=label_pad_token_id,
            dtype=labels[0].dtype
        )
        
        for i, (seq, length) in enumerate(zip(input_ids, seq_lengths)):
            input_ids_padded[i, :length] = seq[:length]
            labels_padded[i, :length] = labels[i][:length]
        
        attention_mask = (input_ids_padded != tokenizer.pad_token_id).long()
        
        if not is_encoder_decoder:
            return {
                'input_ids': input_ids_padded[:, :-1],
                'attention_mask': attention_mask[:, :-1],
                'labels': labels_padded[:, 1:]
            }
        
        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask,
            'labels': labels_padded
        }

class LogitsDecoder:
    def __init__(self, config, tokenizer, temperature=1.0):
        self.config = config
        self.tokenizer = tokenizer
        self.default_temp = max(temperature, 1e-5)
        self.default_top_k = getattr(config, 'top_k', 0)
        self.default_top_p = getattr(config, 'top_p', 1.0)

    def decode_logits(
        self,
        logits: torch.Tensor,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True
    ) -> torch.Tensor:
        temperature = max(temperature if temperature is not None
                            else self.default_temp, 1e-5)
        top_k = top_k if top_k is not None else self.default_top_k
        top_p = top_p if top_p is not None else self.default_top_p

        logits = logits.float()
        
        processors = []
        if repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        
        if do_sample:
            if temperature != 1.0:
                processors.append(TemperatureLogitsWarper(temperature))
            if top_k > 0:
                processors.append(TopKLogitsWarper(top_k))
            if top_p < 1.0:
                processors.append(TopPLogitsWarper(top_p))

        if processors:
            logits = LogitsProcessorList(processors)(None, logits)

        if not do_sample:
            return torch.argmax(logits, dim=-1)
        
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

def clear_directory(directory, include_subdirectories=True):
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path) and include_subdirectories:
                shutil.rmtree(item_path)

def generate_special_token_vocab(min_syl=2, max_syl=2):
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
    if 0 <= index < len(SPECIAL_TOKEN_VOCAB):
        return f'<{SPECIAL_TOKEN_VOCAB[index]}>'
    else:
        return None

SPECIAL_TOKEN_VOCAB = generate_special_token_vocab()

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

                if len(self.ranks) == self.num_nodes - 1:
                    info("All ranks assigned. Shutting down node rank coordinator...")
                    Thread(target=self.graceful_shutdown, daemon=True).start()

                return {'node_rank': assigned_rank}

    def graceful_shutdown(self):
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

def get_fabric_config(dist=False, precision=cfg.precision):
    if dist or get_num_devices() > 1:
        return {'devices': 'auto'}
    else:
        return {
            'devices': 'auto',
            'precision': precision
        }

def set_dist_config(
        config: dict,
        main_addr: Optional[str] = None,
        main_port: Optional[int] = None,
        num_nodes: Optional[int] = None
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

        os.environ['NODE_RANK'] = str(node_rank)
        os.environ['MASTER_ADDR'] = dist_config['main_addr']
        os.environ['MASTER_PORT'] = str(dist_config['main_port'])
        config['fabric_config'].update({'num_nodes': dist_config['num_nodes']})
    
    strategy_name = dist_config['strategy'].lower()
    valid_strategies = {'deepspeed', 'ddp', 'fsdp'}
    if strategy_name not in valid_strategies:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Valid options: {valid_strategies}")
    
    if strategy_name == 'deepspeed':
        from lightning.fabric.strategies import DeepSpeedStrategy
        strategy_config = dist_config.get('deepspeed', {})
        strategy = DeepSpeedStrategy(config=strategy_config)
    else:
        from lightning.fabric.strategies import Strategy
        strategy = Strategy.strategy_from_name(strategy_name)

    config['fabric_config']['strategy'] = strategy

def get_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    is_encoder_decoder: bool = False,
    shift_labels: bool = True,
    ignore_index: int = -100
) -> torch.Tensor:
    if is_encoder_decoder:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=ignore_index
        )
    else:
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=ignore_index
        )
    
    return loss

def get_trainer(model):
    if not cfg.sft and cfg.po == 'SimPO':
        from simpo.simpo_config import SimPOConfig as config
        from simpo.simpo_trainer import SimPOTrainer as trainer
    elif not cfg.sft and cfg.po == 'NCA':
        from nca.nca_config import NCAConfig as config
        from nca.nca_trainer import NCATrainer as trainer
    else:
        config = None
        trainer = None
    
    if trainer:
        from alignment import H4ArgumentParser

        assert version.parse(transformers.__version__) >= version.parse("4.36.0"), \
            f"transformers version 4.36.0 or higher is required, but found {transformers.__version__}. "
        
        parser = H4ArgumentParser((config,))
        training_args = parser.parse(cfg.po_conf_file)
        training_args.max_length = cfg.max_length
        training_args.gradient_checkpointing = False
        training_args.max_prompt_length = cfg.max_prompt_length
        training_args.remove_unused_columns = cfg.remove_unused_columns

        return trainer(
            model=model,
            args=training_args,
            processing_class=model.tokenizer
        )

def generate_messages(category, vocab):
    messages = []
    for entry in TEMPLATE[category]['messages']:
        role = entry['role']
        assert role in ['user', 'system', 'assistant']
        head = entry['content'].get('head', '').strip()
        body = entry['content'].get('body', {})
        if head or body:
            content = [head] if head else []
            for i in body:
                if i in vocab:
                    content.append(f'{i}: {body[i].strip()}')
            messages.append({
                'role': role,
                'content': '\n'.join(content)
            })
    return messages

def get_initial_prompt():
    if cfg.use_initial_prompt:
        if cfg.backend_strategy == 'hint':
            return generate_messages('hint', cfg.backend_vocab[cfg.backend_hint_category])

def get_loss(
    model,
    fabric,
    batch,
    trainer=None,
    label_pad_token_id=LABEL_PAD_TOKEN_ID
):
    is_encoder_decoder = model.config.is_encoder_decoder
    train_eval = 'train' if model.training else 'eval'
    if SHOW_OUTPUT:
        logits_decoder = LogitsDecoder(model.config, model.tokenizer, model.temperature)
    
    if trainer:
        loss, train_metrics = trainer.get_batch_loss_metrics(model, batch, train_eval=train_eval)
        trainer.store_metrics(metrics=train_metrics, train_eval=train_eval)
    else:
        labels = batch['labels']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        if SHOW_INPUT and fabric.is_global_zero:
            for i in range(min(2, len(input_ids))):
                show_input(f"Batch {i}: {model.tokenizer.decode(input_ids[i])}")

        if SHOW_LABELS and fabric.is_global_zero:
            for i in range(min(2, len(labels))):
                valid_ids = [tid for tid in labels[i] if tid != label_pad_token_id]
                show_labels(f"Batch {i}: {model.tokenizer.decode(valid_ids)}")
        
        if model.has_anno:
            outputs = model.annotate(input_ids=input_ids, input_labels=labels, return_loss=True)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        output_logits = outputs['logits']
        
        if SHOW_OUTPUT and fabric.is_global_zero:
            for i in range(min(2, len(output_logits))):
                output_ids = logits_decoder.decode_logits(output_logits[i])
                show_output(f"Batch {i}: {model.tokenizer.decode(output_ids)}")
        
        if model.training and not output_logits.requires_grad:
            raise RuntimeError("Model outputs are not tracking gradients")

        lm_loss = get_lm_loss(
            logits=output_logits,
            labels=labels,
            ignore_index=label_pad_token_id,
            is_encoder_decoder=is_encoder_decoder
        )

        if model.mem_coef:
            if model.has_anno:
                mem_loss = outputs['losses']
            else:
                mem_loss = model.mem.compute_losses(outputs)['total_loss']
            loss = model.lm_coef * lm_loss + model.mem_coef * mem_loss
        else:
            loss = lm_loss

        if model.training and not loss.requires_grad:
            raise RuntimeError("Loss tensor has no gradient connection")

    if model.training:
        fabric.backward(loss)
    
    return loss

def validate_batch(batch):
    if not batch:
        return False
    
    batch_sizes = []
    sequence_keys = ['input_ids', 'labels', 'attention_mask']
    
    for key in sequence_keys:
        if key not in batch:
            continue
            
        tensor = batch[key]
        
        if tensor is None:
            return False
        
        if hasattr(tensor, 'numel'):
            if tensor.numel() == 0:
                return False
            
            if hasattr(tensor, 'shape'):
                for dim in tensor.shape:
                    if dim == 0:
                        return False
                
                if len(tensor.shape) >= 1:
                    batch_sizes.append(tensor.shape[0])
        elif hasattr(tensor, '__len__'):
            if len(tensor) == 0:
                return False
            
            if len(tensor) > 0 and hasattr(tensor[0], '__len__'):
                for seq in tensor:
                    if len(seq) == 0:
                        return False
            
            batch_sizes.append(len(tensor))
        else:
            return False
    
    if batch_sizes and len(set(batch_sizes)) > 1:
        return False
    
    return True

def md_train(
    model,
    loader,
    fabric,
    num_examples=-1,
    log_dir=None,
    log_interval=1,
    trainer=None,
    label_pad_token_id=LABEL_PAD_TOKEN_ID
):
    model.train()
    
    if trainer is None:
        trainer = get_trainer(model)
    
    metrics = {
        'steps': 0,
        'total_loss': 0.0,
        'skipped_batches': 0
    }

    summary_writer = SummaryWriter(log_dir) if log_dir else None
    
    if num_examples > 0:
        max_steps = min(num_examples, len(loader))
        loader_iter = iter(islice(loader, max_steps))
    else:
        max_steps = len(loader)
        loader_iter = iter(loader)
    
    pbar = tqdm(
        loader_iter,
        desc=f'Training',
        disable=not fabric.is_global_zero,
        dynamic_ncols=True,
        total=max_steps
    )
    
    for batch in pbar:
        if not validate_batch(batch):
            metrics['skipped_batches'] += 1
            pbar.set_description(f'Training (skipped: {metrics["skipped_batches"]})')
            continue

        model.step()
        loss = get_loss(model, fabric, batch, trainer, label_pad_token_id)
        
        metrics['steps'] += 1
        metrics['total_loss'] += loss.item()
        
        if summary_writer:
            sample_progress = int((metrics['steps'] / max_steps) * 100)
            if sample_progress % log_interval == 0:
                summary_writer.add_scalar("loss/steps", loss.item(), metrics['steps'])
    
    if trainer is None:
        if metrics['steps'] > 0:
            metrics['total_loss'] /= metrics['steps']
    else:
        train_metrics = trainer.get_metrics()['train']
        for key, val in train_metrics.items():
            metrics[key] = torch.tensor(val).mean().item()

    pbar.close()

    if summary_writer:
        summary_writer.close()

    return metrics

def md_validate(
    model, 
    loader,
    fabric,
    num_examples=-1,
    log_dir=None,
    log_interval=1,
    trainer=None,
    label_pad_token_id=LABEL_PAD_TOKEN_ID
) -> dict:
    if loader is None:
        return {}
    
    model.eval()

    if trainer is None:
        trainer = get_trainer(model)

    metrics = {
        'steps': 0,
        'avg_loss': 0.0,
        'total_loss': 0.0,
        'skipped_batches': 0
    }

    summary_writer = SummaryWriter(log_dir) if log_dir else None
    
    if num_examples > 0:
        max_steps = min(num_examples, len(loader))
        loader_iter = iter(islice(loader, max_steps))
    else:
        max_steps = len(loader)
        loader_iter = iter(loader)
    
    pbar = tqdm(
        loader_iter,
        desc='Validating',
        disable=not fabric.is_global_zero,
        dynamic_ncols=True,
        total=max_steps
    )

    with torch.no_grad():
        for batch in pbar:
            if not validate_batch(batch):
                metrics['skipped_batches'] += 1
                pbar.set_description(f'Validating (skipped: {metrics["skipped_batches"]})')
                continue
            
            loss = get_loss(model, fabric, batch, trainer, label_pad_token_id)
            
            metrics['steps'] += 1
            metrics['total_loss'] += loss.item()

            if summary_writer:
                sample_progress = int((metrics['steps'] / max_steps) * 100)
                if sample_progress % log_interval == 0:
                    summary_writer.add_scalar("loss/steps", loss.item(), metrics['steps'])

    if trainer is None:
        gathered_loss = fabric.all_gather(torch.tensor(metrics['total_loss'])).sum()
        gathered_examples = fabric.all_gather(torch.tensor(metrics['steps'])).sum()
        if fabric.is_global_zero and gathered_examples > 0:
            avg_loss = gathered_loss / gathered_examples
            metrics['avg_loss'] = avg_loss
            print(f"Average Loss: {avg_loss:.4f} | Examples: {gathered_examples}")

        if metrics['steps'] > 0:
            metrics['total_loss'] /= metrics['steps']
    else:
        eval_metrics = trainer.get_metrics()['eval']
        for key, val in eval_metrics.items():
            metrics[key] = torch.tensor(val).mean().item()
            gathered_vals = fabric.all_gather(torch.tensor(val))
            if fabric.is_global_zero:
                metrics[f'avg_{key}'] = gathered_vals.mean().item()
                print(f"avg_{key}: {metrics[f'avg_{key}']}")

    pbar.close()

    if summary_writer:
        summary_writer.close()
    
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
                    info(f"node_rank: {response.json()['node_rank']}")
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

def get_num_devices():
    if CUDAAccelerator.is_available():
        return CUDAAccelerator.auto_device_count()
    else:
        return 1

def get_strategy(precision):
    if get_num_devices() > 1:
        if STRATEGY == 'ddp':
            from lightning.fabric.strategies import DDPStrategy
            return DDPStrategy(find_unused_parameters=True)
        elif STRATEGY == 'deepspeed':
            from lightning.fabric.strategies import DeepSpeedStrategy
            strategy_config = {
                'zero_optimization': {
                    'stage': 0
                },
                'activation_checkpointing': {
                    'enable': True,
                    'use_reentrant': True
                }
            }
            return DeepSpeedStrategy(config=strategy_config)
