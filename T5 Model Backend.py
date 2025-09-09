from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi import FastAPI
import torch
import re
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
t5_model.eval()
app = FastAPI()
class clueSolver(BaseModel):
    clue: str
    answer_length: int
    answer_letters: str
@app.post("/solver_answer")
def solveClue(clue_info: clueSolver):
    t5_prompt = f"answer question: {clue_info.clue} length {clue_info.answer_length} pattern {clue_info.answer_letters}"
    t5_input_tokens = t5_tokenizer.encode(t5_prompt, return_tensors = "pt")
    with torch.no_grad():
        t5_outputs = t5_model.generate(
            t5_input_tokens,
            max_length = clue_info.answer_length + 5,
            min_length = 2,
            num_beams = 25,
            num_return_sequences = 20,
            return_dict_in_generate = True,
            output_scores = True
        )
    t5_output_sequences = t5_outputs.sequences
    t5_output_scores = t5_outputs.scores
    