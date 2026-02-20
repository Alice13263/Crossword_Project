#Importing the dataset & tokenizers
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM)
import torch
#Supervised Fine Tuning trains the model to perform a specific task (here to generate clues)
from trl import SFTTrainer, SFTConfig
#LoRA allows me to train the model on a small scale that doesn't require too many resources
from peft import LoraConfig
#Using the same dataset as the clue solver for coherency in responses
crossword_dataset = load_dataset("azugarini/clue-instruct")
#Using TinyLlama to be able to fine tune the model on my device without lots of processing power
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
clue_generate_tokenizer = AutoTokenizer.from_pretrained(model_name)
clue_generate_tokenizer.pad_token = clue_generate_tokenizer.eos_token
training_crossword_dataset = crossword_dataset["train"]
#Formats all the clues and answers in the dataset to fit how I will be using the model later
def clue_format(example):
    formatted_clues = example["clues"]
    formatted_answer = example["keyword"]
    examples = []
    for answer, clues in zip(formatted_answer, formatted_clues):
        if answer is None:
            continue
        answer = str(answer).upper()
        if isinstance(clues,str):
            formatted_clues_list = [clues]
        elif isinstance(clues,list):
            formatted_clues_list = clues
        else:
            continue
        for clue in formatted_clues_list:
            clue = str(clue).strip()
            if clue is None or len(clue) < 3:
                continue
            #TinyLlama has a specific format for receiving instructions, so this must be followed here
            example_prompt = f"<s>[INST] Write a crossword clue for this answer: {answer} [/INST] {clue}</s>"
            examples.append(example_prompt)
    return {"text": examples}
training_crossword_dataset = training_crossword_dataset.map(clue_format, batched = True, remove_columns = training_crossword_dataset.column_names, batch_size = 100)
#Loads the model in a bit format, which uses much less memory and can be run on my device
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map = {"": "cpu"},
    torch_dtype = torch.float32
)
#Fine tunes the model using LoRA to reduce the processing needed when training
lora_fine_tuning = LoraConfig(
    r = 16,
    lora_alpha = 32,
    target_modules = ["q_proj", "v_proj"],
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
)
model.config.pad_token_id = clue_generate_tokenizer.eos_token_id
#Setting the arguments for the training process - batch sizes are smaller than the clue solver because more needs to be generated and to a higher standard
training = SFTConfig(
    output_dir = "./clueGeneratorFinetuned",
    learning_rate = 2e-4,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    num_train_epochs = 3,
    logging_steps = 25,
    save_strategy = "steps",
    save_steps = 200,
    fp16 = False,
    max_length = 256
)
#Sets the trainer to format the TinyLlama to the specific task (here generating clues)
trainer = SFTTrainer(
    model = model,
    args = training,
    peft_config = lora_fine_tuning,
    train_dataset = training_crossword_dataset,
    processing_class = clue_generate_tokenizer,
)
#Runs the training and saves the results in the directory
trainer.train()
trainer.model.save_pretrained("./clueGeneratorFinetuned")
clue_generate_tokenizer.save_pretrained("./clueGeneratorFinetuned")