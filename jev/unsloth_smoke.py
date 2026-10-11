"""Minimal Unsloth LoRA SFT smoke test, tracked in ClearML."""
from unsloth import FastLanguageModel  # must be imported before trl/transformers

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from jev.tracking import init_task

config = {
    "model_name": "unsloth/Qwen3-0.6B",
    "max_seq_length": 256,
    "lora_r": 8,
    "lora_alpha": 16,
    "max_steps": 10,
    "learning_rate": 2e-4,
    "batch_size": 2,
}
task = init_task("unsloth-smoke-test", config, tags=["setup", "unsloth"])

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["model_name"],
    max_seq_length=config["max_seq_length"],
    load_in_4bit=False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=config["lora_r"],
    lora_alpha=config["lora_alpha"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

rows = [
    ("amount=120.00 country=SG merchant=grocery velocity=1", "approve"),
    ("amount=9800.00 country=NG merchant=crypto velocity=14", "decline"),
    ("amount=450.00 country=US merchant=electronics velocity=3", "review"),
    ("amount=15.50 country=SG merchant=transit velocity=1", "approve"),
] * 8
data = Dataset.from_list(
    [{"text": f"Transaction: {x}\nDecision: {y}{tokenizer.eos_token}"} for x, y in rows]
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=data,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        per_device_train_batch_size=config["batch_size"],
        max_steps=config["max_steps"],
        learning_rate=config["learning_rate"],
        logging_steps=1,
        output_dir="outputs/unsloth-smoke",
        report_to="none",
    ),
)
stats = trainer.train()
print("train_loss", stats.training_loss)
task.get_logger().report_single_value("train_loss", stats.training_loss)
task.close()
