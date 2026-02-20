#LoRA configurations do not contain the actual model, so my finetuned version must be merged with the base model
#This file only runs once
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
finetuned_model = "./clueGeneratorFinetuned/checkpoint-30200"
original_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
merged_model = "./clueGeneratorMerged"
tokenizer = AutoTokenizer.from_pretrained(original_model)
#Adding the base model to the finetuned version
base_model = AutoModelForCausalLM.from_pretrained(
    original_model,
    device_map = "cpu",
    torch_dtype = torch.float16
)
#Saving the new model
new_model = PeftModel.from_pretrained(base_model, finetuned_model)
new_model = new_model.merge_and_unload()
new_model.save_pretrained(merged_model)
tokenizer.save_pretrained(merged_model)
