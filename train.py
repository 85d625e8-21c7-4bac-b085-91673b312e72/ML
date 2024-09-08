import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import numpy as np
import wandb
os.environ["WANDB_PROJECT"]="Function-Calling"
os.environ["WANDB_NAME"]="RU"
os.environ["WANDB_LOG_MODEL"]="true"
os.environ["WANDB_WATCH"]="false"
import torch
from datasets import load_dataset, Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


REVISION_NAME = "_20"
DATA_CACHE_DIR = "data"
MODEL_NAME = "AnatoliiPotapov/T-lite-instruct-0.1"  
MODEL_CACHE_DIR = "cache"
HF_TOKEN = ""
os.environ['HF_TOKEN'] = HF_TOKEN
MAX_LENGTH = 2300
STEP_SIZE = 1
TRAIN_BS = 2
TRAIN_EPOCHS = 1
LORA_R = 64
LORA_ALPHA = 64
TARGET_MODULES = [
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]
device_map="FSDP"

# Llama
BOS_TOKEN = "<|begin_of_text|>"
EOS_TOKEN = "<|end_of_text|>"
END_TURN = "<|eot_id|>"
USER_TURN = "<|start_header_id|>user<|end_header_id|>\n\n"
MODEL_TURN = "<|start_header_id|>assistant<|end_header_id|>\n\n"
BEGIN_FUNCTION_CALL = "Ты - полезный помощник, имеющий доступ к следующим функциям. Используйте их при необходимости - "
END_FUNCTION_CALL = "\n\n"
FUNCTION_CALL_PREFIX = "Function call: "
FUNCTION_RESPONSE_PREFIX = "Function response: "


def reformat_text_input(example):
    if example['is_function_call']:
        text = f"""{BOS_TOKEN}{USER_TURN}{BEGIN_FUNCTION_CALL}{example['function_description'].strip()}{END_FUNCTION_CALL}{example['query'].strip()}{END_TURN}{MODEL_TURN}{FUNCTION_CALL_PREFIX}{example['response'].strip()}{ENDTURN_AND_EOS_TOKEN}"""
    elif example['is_function_response']:
        text = f"""{BOS_TOKEN}{USER_TURN}{BEGIN_FUNCTION_CALL}{example['function_description'].strip()}{END_FUNCTION_CALL}{FUNCTION_RESPONSE_PREFIX}{example['query'].strip()}{END_TURN}{MODEL_TURN}{example['response'].strip()}{ENDTURN_AND_EOS_TOKEN}"""
    else:
        text = f"""{BOS_TOKEN}{USER_TURN}{BEGIN_FUNCTION_CALL}{example['function_description'].strip()}{END_FUNCTION_CALL}{example['query'].strip()}{END_TURN}{MODEL_TURN}{example['response'].strip()}{ENDTURN_AND_EOS_TOKEN}"""

    example['text'] = text

    return example


def main():
    # Prepare model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        token=HF_TOKEN,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        cache_dir=MODEL_CACHE_DIR,
        use_cache=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    # Prepare dataset
    dataset = load_dataset("DiTy/function-calling", "ru", cache_dir=DATA_CACHE_DIR, token=HF_TOKEN)  # token=HF_TOKEN
    print(dataset)
    dataset = dataset.map(reformat_text_input)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=MODEL_TURN,
        tokenizer=tokenizer,
        mlm=False,
    )

    # ## FOR TEST ##
    # dataset_vvalid = Dataset.from_dict(dataset['valid'][:1])

    # Prepare metric
    # meteor = evaluate.load('meteor', cache_dir=MODEL_CACHE_DIR)
    

    # def preprocess_logits_for_metrics(logits, labels):
    #     pred_ids = torch.argmax(logits[0], dim=-1)
        
    #     return pred_ids, labels


    # def compute_metrics(eval_pred):
    #     predictions, labels = eval_pred
    #     print(type(predictions), type(labels))
    #     # logits = torch.from_numpy(logits)
    #     labels = torch.from_numpy(labels)
    #     labels[labels == -100] = tokenizer.pad_token_id
    #     labels[labels == tokenizer.eos_token_id] = tokenizer.pad_token_id

    #     predictions = predictions[0]
    #     # predictions = np.argmax(logits, axis=-1)  # [B; S]
    #     no_pad_mask = labels != tokenizer.pad_token_id  # != pad token
    #     # print("PREDICTIONS: ", predictions, '\n\n', "LABELS: ", labels, '\n\n', "MASK_FROM_LABELS: ", no_pad_mask, '\n\n', )

    #     # Cut from start labels
    #     start_ids = no_pad_mask.to(torch.int8).argmax(keepdim=True, dim=1)
    #     n_rows, n_cols = predictions.size()
    #     from_start_ids_mask = torch.arange(n_cols).expand_as(predictions) >= start_ids
    #     predictions = torch.where(from_start_ids_mask, predictions, tokenizer.pad_token_id)
    #     # print("PREDICTIONS AFTER CUT START: ", predictions)

    #     # Cut before EOS tokens in preds 
    #     eos_mask = predictions == tokenizer.convert_tokens_to_ids(END_TURN)  # (predictions == tokenizer.eos_token_id) | (== eos token
    #     # print("END MASK: ", eos_mask)
    #     end_ids = eos_mask.to(torch.int8).argmax(keepdim=True, dim=1)
    #     # print("END IDS: ", end_ids)
    #     n_rows, n_cols = predictions.size()
    #     before_eos_mask = torch.arange(n_cols).expand_as(predictions) < end_ids
    #     predictions = torch.where(before_eos_mask, predictions, tokenizer.pad_token_id)
    #     # print("PREDICTIONS AFTER FULL CUT: ", predictions)

    #     text_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #     text_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
    #     # print("TEXT_PREDICTIONS", text_predictions)
    #     # print("TEXT_LABELS", text_labels)

    #     # Compute metric
    #     results = meteor.compute(
    #         predictions=text_predictions, 
    #         references=text_labels
    #     )
        
    #     return results


    # Prepare Lora
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=0.1,
        bias="none",
    )

    # Args
    training_arguments = TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir="function-calling-checkpoints/",
        overwrite_output_dir=True,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        eval_strategy="steps",
        eval_steps=STEP_SIZE,
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        optim="adafactor",
        save_steps=STEP_SIZE,
        save_total_limit=10,
        logging_steps=STEP_SIZE,
        learning_rate=1e-4,
        bf16=True,
        bf16_full_eval=True,
        max_grad_norm=1.0,
        lr_scheduler_type="constant",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs = {"use_reentrant": True},
        group_by_length=True,
        report_to="wandb",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        dataset_text_field="text",
        peft_config=lora_config,
        max_seq_length=MAX_LENGTH,
        data_collator=collator,
        packing=False,
        args=training_arguments,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=15)],
    )

    # handle PEFT+FSDP case
    trainer.model.print_trainable_parameters()
    if getattr(trainer.accelerator.state, "fsdp_plugin", None):
        from peft.utils.other import fsdp_auto_wrap_policy

        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)


    # Training
    trainer.train()


    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(f"/model-lora-1ep/{REVISION_NAME}")


    # Finish logging
    wandb.finish()


if __name__ == "__main__":
    main()
