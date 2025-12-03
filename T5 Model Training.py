from datasets import load_dataset
from transformers import (T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, TrainingArguments, Trainer)
crossword_dataset = load_dataset("azugarini/clue-instruct")
model = T5ForConditionalGeneration.from_pretrained("t5-base")
solver_tokenizer = T5Tokenizer.from_pretrained("t5-base")
def setup(data):
    clue = f"Clue: {data["clues"]}"
    answer = data["keyword"]
    encode_tokens = solver_tokenizer(clue, truncation = True, padding = "max_length", max_length = 64)
    encode_outputs = solver_tokenizer(answer, truncation = True, padding = "max_length", max_length = 64)
    encode_tokens["labels"] = encode_outputs["input_ids"]
    return encode_tokens
     
tokenized_crossword_dataset = crossword_dataset.map(setup, batched = False, remove_columns = crossword_dataset["train"].column_names)
training = TrainingArguments(
    output_dir = "./crosswordSolverFinetuned",
    learning_rate = 3e-4,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
    gradient_accumulation_steps = 4,
    num_train_epochs = 3,
    weight_decay = 0.01,
    save_total_limit = 2,
    logging_steps = 50,
    push_to_hub = False 
)
collated_data = DataCollatorForSeq2Seq(solver_tokenizer, model = model)
trainer = Trainer(
    model = model,
    args = training,
    train_dataset = tokenized_crossword_dataset["train"],
    eval_dataset = tokenized_crossword_dataset["test"],
    tokenizer = solver_tokenizer,
    data_collator = collated_data,
)
trainer.train()
model.save_pretrained("./crosswordSolverFinetuned")
solver_tokenizer.save_pretrained("./crosswordSolverFinetuned")
