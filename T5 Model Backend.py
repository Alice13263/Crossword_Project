from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi import FastAPI
from fastapi.responses import JSONResponse
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
    t5_prompt = f"answer question: {clue_info.clue} (length {clue_info.answer_length}) (pattern {clue_info.answer_letters})"
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
    t5_answers = []
    for i in t5_output_sequences:
        t5_answer_text = t5_tokenizer.decode(i, skip_special_tokens = True).strip()
        sum_log_probs = 0
        tokens_count = 0
        for j in range (len(i)-1):
            t5_tensor = t5_output_scores[j]
            next_token = i[j+1]
            current_log_prob = torch.log_softmax(t5_tensor, dim = 1)[0, next_token].item()
            sum_log_probs += current_log_prob
            tokens_count += 1
        average_log_prob = sum_log_probs / tokens_count
        confidence_level = torch.exp(torch.tensor(average_log_prob)).item()*100
        confidence_level = round(confidence_level, 2)
        t5_answers.append({"answer": t5_answer_text.lower(), "percentage": confidence_level})
    # Sets detect duplicates, so all answers will be passed through here, and then added to another list once checked
    t5_duplicates = set()
    t5_answers_unique = []
    for i in t5_answers:
        if i["answer"] not in t5_duplicates:
            t5_duplicates.add(i["answer"])
            t5_answers_unique.append(i)
    # Regex does not accept _ as a wildcard letter, so all instances will be replaced with ., so that the list can be ordered by matching patterns
    matching_pattern_regex = clue_info.answer_letters.replace("_", ".")
    t5_answers_ranked = []
    for i in t5_answers_unique:
        answer = i["answer"]
        confidence = i["percentage"]
        length_match = True if (len(answer) == clue_info.answer_length) else False
        regex_match = True if (re.fullmatch(matching_pattern_regex, answer, re.IGNORECASE)) else False
        t5_answers_ranked.append({"answer": answer, "percentage": confidence, "regex_match": regex_match, "length_match": length_match})
    # Reverse = True is used to flip the list, as False == 0, comes before True == 1, when I actually need the True values to be first. Also need confidence levels to start with the highest, so descending order
    t5_answers_ordered = sorted(t5_answers_ranked, key = lambda ans: (ans["regex_match"], ans["length_match"], ans["percentage"]), reverse = True)
    t5_top_10 = t5_answers_ordered[:10]
    return JSONResponse({"results": t5_top_10})