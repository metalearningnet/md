# Modified from https://github.com/huggingface/trl/blob/2f726ce4e88a99b5d20eca3b5482954851d91ef6/trl/trainer/dpo_trainer.py
# strongly recommend comparing with ./trl/trl/trainer/dpo_trainer.py line-by-line to identify code changes.

# Original License
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import random
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl.trainer.utils import disable_dropout_in_model, pad_to_length
from nca.nca_utils import NCADataCollatorWithPadding

is_peft_available = False
is_wandb_available = False

if is_peft_available:
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available:
    import wandb

class NCATrainer(Trainer):
    r"""
    Initialize NCATrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        beta (`float`, defaults to 0.1):
            The beta factor in NCA loss. Higher beta means less divergence from the initial policy.
        loss_type (`str`, defaults to `"sigmoid"`):
            The type of NCA loss to use. Either `"NCA"` or `"InfoNCA"`.
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`NCADataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value. This argument is required if you want to use the default data collator.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        processing_class (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_target_length (`int`, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        model_init_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
        ref_model_init_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the ref model from a string
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        temperature_alpha: float = 1e-3, 
        loss_type: Literal["InfoNCA", "NCA"] = "NCA",
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
    ):
        tokenizer = processing_class

        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the NCATrainer. But your model is already instantiated.")

        if ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_kwargs to the NCATrainer. But your ref_model is already instantiated."
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the NCATrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the NCATrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        if not is_peft_available and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                preprare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    preprare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **preprare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)

        # For models that use gradient_checkpoiting, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval and not is_wandb_available:
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = is_peft_available and isinstance(model, PeftModel)
        self.ref_model = ref_model if ref_model else None
    
        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default NCADataCollatorWithPadding"
                )
            if max_length is None:
                max_length = getattr(args, "max_length")
                if max_length is None:
                    warnings.warn(
                        "When using NCADataCollatorWithPadding, you should set `max_length` in the NCATrainer's init"
                        " it will be set to `512` by default, but you should do it yourself in the future.",
                        UserWarning,
                    )
                    max_length = 512
            if max_prompt_length is None:
                max_prompt_length = getattr(args, "max_prompt_length")
                if max_prompt_length is None:
                    warnings.warn(
                        "When using NCADataCollatorWithPadding, you should set `max_prompt_length` in the NCATrainer's init"
                        " it will be set to `128` by default, but you should do it yourself in the future.",
                        UserWarning,
                    )
                    max_prompt_length = 128

            if max_target_length is None and self.is_encoder_decoder:
                warnings.warn(
                    "When using NCADataCollatorWithPadding with an encoder decoder architecture, you should set `max_target_length` in the NCATrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_target_length = 128

            data_collator = NCADataCollatorWithPadding(
                tokenizer,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
                is_encoder_decoder=self.is_encoder_decoder,
                max_target_length=max_target_length,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                warnings.warn(
                    "When using NCADataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value

        self.beta = beta
        self.temperature_alpha = temperature_alpha
        self.loss_type = loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        if args:
            args.output_dir = 'outputs'
        
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self.processing_class = processing_class
        
        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        if self.ref_model:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:#
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if self.is_encoder_decoder:
            raise NotImplementedError
        else:
            max_length = max(batch["A0_input_ids"].shape[1], batch["A1_input_ids"].shape[1], batch["A2_input_ids"].shape[1], batch["A3_input_ids"].shape[1])
        
        concatenated_batch = {
            "concatenated_input_ids": torch.cat(
                    (
                        pad_to_length(batch["A0_input_ids"], max_length, pad_value=self.padding_value),
                        pad_to_length(batch["A1_input_ids"], max_length, pad_value=self.padding_value),
                        pad_to_length(batch["A2_input_ids"], max_length, pad_value=self.padding_value),
                        pad_to_length(batch["A3_input_ids"], max_length, pad_value=self.padding_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device),
            "concatenated_attention_mask": torch.cat(
                    (
                        pad_to_length(batch["A0_attention_mask"], max_length, pad_value=self.padding_value),
                        pad_to_length(batch["A1_attention_mask"], max_length, pad_value=self.padding_value),
                        pad_to_length(batch["A2_attention_mask"], max_length, pad_value=self.padding_value),
                        pad_to_length(batch["A3_attention_mask"], max_length, pad_value=self.padding_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device),
            "concatenated_labels": torch.cat(
                    (
                        pad_to_length(batch["A0_labels"], max_length, pad_value=self.label_pad_token_id),
                        pad_to_length(batch["A1_labels"], max_length, pad_value=self.label_pad_token_id),
                        pad_to_length(batch["A2_labels"], max_length, pad_value=self.label_pad_token_id),
                        pad_to_length(batch["A3_labels"], max_length, pad_value=self.label_pad_token_id),
                    ),
                    dim=0,
                ).to(self.accelerator.device),        
            }

        if self.is_encoder_decoder:
            raise NotImplementedError

        return concatenated_batch

    def nca_loss(
        self,
        batch,
        policy_A0_logps: torch.FloatTensor,
        policy_A1_logps: torch.FloatTensor,
        policy_A2_logps: torch.FloatTensor,
        policy_A3_logps: torch.FloatTensor,
        reference_A0_logps: torch.FloatTensor,
        reference_A1_logps: torch.FloatTensor,
        reference_A2_logps: torch.FloatTensor,
        reference_A3_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the NCA loss for a batch of policy and reference model log probabilities.
        """
        A0_reward = (policy_A0_logps - reference_A0_logps) * self.beta
        A1_reward = (policy_A1_logps - reference_A1_logps) * self.beta
        A2_reward = (policy_A2_logps - reference_A2_logps) * self.beta
        A3_reward = (policy_A3_logps - reference_A3_logps) * self.beta
        # The definition of temperature_alpha here is different from that in the paper (temperature_alpha = 1 / paper_alpha)
        # rewards = torch.stack([batch["A0_score"],batch["A1_score"],batch["A2_score"],batch["A3_score"]],dim=-1) / self.temperature_alpha #<bz,4>
        # +0.01 here ensures A0 has the highest reward even if r(A1) = r(A0). This is included merely to stay consistent with preference settings. Could be removed.
        rewards = torch.stack([batch["A0_score"] + 0.01, batch["A1_score"], batch["A2_score"], batch["A3_score"]], dim=-1) / self.temperature_alpha #<bz,4>
        softlabel = rewards.softmax(dim=-1) #<bz,4>
        model_rewards = torch.stack([A0_reward, A1_reward, A2_reward, A3_reward], dim=-1) #<bz,4>

        if self.loss_type == "InfoNCA":
            ratio_logits_p = model_rewards.log_softmax(dim=-1)
            losses = - (softlabel * ratio_logits_p).sum(dim=-1)
        elif self.loss_type == "NCA":
            losses = -F.logsigmoid(-model_rewards).mean() - (softlabel * F.logsigmoid(model_rewards)).sum(dim=-1).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['InfoNCA', 'NCA']")

        return losses, A0_reward.detach(), A1_reward.detach(), A2_reward.detach(), A3_reward.detach()

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not self.is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def get_logps(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        total = 4
        results = {}
        for i in range(total):
            required_keys = [f'A{i}_input_ids', f'A{i}_attention_mask', f'A{i}_labels']
            if not all(k in batch for k in required_keys):
                raise ValueError(f"Batch must contain all of {required_keys}")
        
        if self.model.has_anno:
            loss_list = []
            for i in range(total):
                input_ids = batch[f'A{i}_input_ids']
                input_labels = batch[f'A{i}_labels']
                if not self.model.dist:
                    with torch.autocast(model.device.type, dtype=model.dtype):
                        model_out = model.annotate(input_ids, input_labels=input_labels, return_loss=True)
                else:
                    model_out = model.annotate(input_ids, input_labels=input_labels, return_loss=True)
                logits = model_out['logits']
                labels = model_out['labels']
                losses = model_out['losses']
                
                logps = self.get_batch_logps(
                    logits,
                    labels,
                    average_log_prob=True
                )

                results[f'A{i}_logps'] = logps
                results[f'A{i}_logits'] = logits
                
                if losses is not None:
                    loss_list.append(losses)
            
            losses = torch.stack(loss_list).mean() if loss_list else None
        else:
            # Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
            # We do this to avoid doing two forward passes, because it's faster for FSDP.
            concatenated_batch = self.concatenated_inputs(batch)
            len_chosen = batch["A0_labels"].shape[0]

            model_kwargs = {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            } if self.is_encoder_decoder else {}
            
            model_out = model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                **model_kwargs,
            )

            losses = model.skill_memory.compute_losses(model_out)['total_loss']
            all_logits = model_out['logits']
            all_logps = self.get_batch_logps(
                all_logits.to(torch.float32),
                concatenated_batch["concatenated_labels"],
                average_log_prob=False,
            )

            for i in range(total):
                results[f'A{i}_logps'] = all_logps[i * len_chosen:(i + 1) * len_chosen]
                results[f'A{i}_logits'] = all_logits[i * len_chosen:(i + 1) * len_chosen]
            
        logps = [results[f'A{i}_logps'] for i in range(total)]
        logits = [results[f'A{i}_logits'] for i in range(total)]

        return (*logps, *logits, losses)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the NCA loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        # TODO support arbitrary K option
        with torch.no_grad():
            m = model if self.ref_model is None else self.ref_model
            ref_outputs = self.get_logps(m, batch)
            reference_A0_logps, reference_A1_logps, reference_A2_logps, reference_A3_logps = ref_outputs[:4]
        
        policy_outputs = self.get_logps(model, batch)
        policy_A0_logps, policy_A1_logps, policy_A2_logps, policy_A3_logps = policy_outputs[:4]

        skill_loss = policy_outputs[-1] if policy_outputs[-1] is not None else torch.tensor(0.0, device=model.device, requires_grad=True)

        losses, A0_rewards, A1_rewards, A2_rewards, A3_rewards = self.nca_loss(
            batch,
            policy_A0_logps,
            policy_A1_logps,
            policy_A2_logps,
            policy_A3_logps,
            reference_A0_logps,
            reference_A1_logps,
            reference_A2_logps,
            reference_A3_logps,
        )

        reward_accuracies = ((A0_rewards > A1_rewards).float() + 
                        (A0_rewards > A2_rewards).float() + 
                        (A0_rewards > A3_rewards).float()) / 3.0

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/A0"] = A0_rewards.cpu().mean()
        metrics[f"{prefix}rewards/A1"] = A1_rewards.cpu().mean()
        metrics[f"{prefix}rewards/A2"] = A2_rewards.cpu().mean()
        metrics[f"{prefix}rewards/A3"] = A3_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (A0_rewards - (A1_rewards + A2_rewards + A3_rewards) / 3.0).cpu().mean()
        metrics[f"{prefix}logps/A0"] = policy_A0_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/A1"] = policy_A1_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/A2"] = policy_A2_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/A3"] = policy_A3_logps.detach().cpu().mean()

        lm_loss = losses.mean()
        total_loss = model.lm_coef * lm_loss + model.skill_coef * skill_loss

        if torch.is_grad_enabled():
            assert total_loss.requires_grad, "Total loss doesn't require gradients!"
        
        return total_loss, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for NCADataCollatorWithPadding, and you passed a datacollator that is different than "
                "NCADataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        
        return loss

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        policy_output = model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.max_length,
            do_sample=True,
            pad_token_id=self.processing_class.pad_token_id,
        )

        if self.ref_model is None:
            with self.accelerator.unwrap_model(self.model).disable_adapter():
                reference_output = self.model.generate(
                    batch["prompt_input_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_length=self.max_length,
                    do_sample=True,
                    pad_token_id=self.processing_class.pad_token_id,
                )
        else:
            reference_output = self.ref_model.generate(
                batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.processing_class.pad_token_id,
            )

        policy_output = pad_to_length(policy_output, self.max_length, self.processing_class.pad_token_id)
        policy_output_decoded = self.processing_class.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.processing_class.pad_token_id)
        reference_output_decoded = self.processing_class.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for NCADataCollatorWithPadding, and you passed a datacollator that is different than "
                "NCADataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def get_metrics(self):
        return self._stored_metrics
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                            for prompt, pol, ref in zip(
                                random_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)
