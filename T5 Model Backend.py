from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import re
import uvicorn
t5_model = T5ForConditionalGeneration.from_pretrained("./crosswordSolverFinetuned")
t5_tokenizer = T5Tokenizer.from_pretrained("./crosswordSolverFinetuned")
t5_model.eval()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class clueSolver(BaseModel):
    clue_valid: str
    number_letters_valid: int
    given_letters_valid: str
@app.post("/solverAnswer")
async def solveClue(clue_info: clueSolver):
    t5_prompt = (f"Clue: {clue_info.clue_valid}\n"f"Keyword:\n"f"Answer:")
    t5_input_tokens = t5_tokenizer.encode(t5_prompt, return_tensors = "pt")
    with torch.no_grad():
        t5_outputs = t5_model.generate(
            t5_input_tokens,
            min_length = 2,
            max_length = clue_info.number_letters_valid + 5,
            num_beams = 25,
            num_return_sequences = 20,
            return_dict_in_generate = True,
            output_scores = True
        )
    t5_output_sequences = t5_outputs.sequences
    t5_output_scores = t5_outputs.scores
    t5_answers = []
    for seq in t5_output_sequences:
        t5_answer_text = t5_tokenizer.decode(seq, skip_special_tokens = True).strip()
        sum_log_probs = 0
        tokens_count = 0
        for token in range (len(seq)-1):
            t5_tensor = t5_output_scores[token]
            next_token = seq[token+1]
            current_log_prob = torch.log_softmax(t5_tensor, dim = 1)[0, next_token].item()
            sum_log_probs += current_log_prob
            tokens_count += 1
        average_log_prob = sum_log_probs / tokens_count
        confidence_level = round(torch.exp(torch.tensor(average_log_prob)).item()*100,2)
        t5_answers.append({"answer": t5_answer_text.lower(), "percentage": confidence_level})
    t5_answers_unique = []
    for answer in t5_answers:
        if answer["answer"] not in t5_answers_unique:
            t5_answers_unique.append(answer)
    # Regular expression does not accept _ as a wildcard letter, so all instances will be replaced with ., so that the list can be ordered by matching patterns
    matching_pattern = clue_info.given_letters_valid.replace("_", ".")
    t5_answers_ranked = []
    for unique_answer in t5_answers_unique:
        answer = unique_answer["answer"]
        confidence = unique_answer["percentage"]
        if (len(answer) == clue_info.number_letters_valid):
            length_match = True
        else:
            length_match = False
        if (re.fullmatch(matching_pattern, answer, re.IGNORECASE)):
            pattern_match = True
        else:
            pattern_match = False
        t5_answers_ranked.append({"answer": answer, "percentage": confidence, "pattern_match": pattern_match, "length_match": length_match})
    # Reverse = True is used to flip the list, as False == 0, comes before True == 1, when I actually need the True values to be first. Also need confidence levels to start with the highest, so descending order
    t5_answers_ordered = sorted(t5_answers_ranked, key = lambda ans: (ans["percentage"], ans["pattern_match"], ans["length_match"]), reverse = True)[:10]
    return JSONResponse({"results": t5_answers_ordered})
if __name__ == "__main__":
    uvicorn.run("T5 Model Backend:app", host="127.0.0.1", port=8000, reload=True)