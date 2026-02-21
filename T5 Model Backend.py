from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import re
import uvicorn
#Using .env to access the database, to keep it more secure
import os
from dotenv import load_dotenv
#Importing supabase so that the database can be accessed
from supabase import create_client, Client
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
#Connecting to the supabase database
load_dotenv()
supabase_url = os.getenv("supabase_url")
supabase_service_key = os.getenv("supabase_service_key")
supabase: Client = create_client(supabase_url, supabase_service_key)
class clueSolver(BaseModel):
    clue_valid: str
    number_letters_valid: int
    given_letters_valid: str
    #Username is now added so that it can be searched in the database
    username: str
@app.post("/solverAnswer")
async def solveClue(clue_info: clueSolver):
    #Instead of calling the model immediately, the database will be searched in case the clue has been solved before
    clue_text = clue_info.clue_valid.strip()
    answer_length = clue_info.number_letters_valid
    pattern = clue_info.given_letters_valid.strip()
    clue_username = clue_info.username.strip()
    #Checking the database to see if the clue has been solved before, as these answers can be used instead
    check_database = supabase.table("cluesolver")\
        .select("cluesolveid")\
        .eq("clueentered", clue_text)\
        .eq("totalletters", answer_length)\
        .execute()
    if check_database.data:
        clue_id = check_database.data[0]["cluesolveid"]
        #Finding the player's user ID using their username
        user_id = supabase.table("users")\
            .select("userid")\
            .eq("username", clue_username)\
            .execute()
        user_id_value = user_id.data[0]["userid"]
        #If this player searched this clue before, the same answers can be given
        user_check = supabase.table("user_cluesolver")\
            .select("*")\
            .eq("userid", user_id_value)\
            .eq("cluesolveid", clue_id)\
            .execute()
        #If it was a different player that searched this clue, then give the player the answers, but add to the database that they also searched for it
        if not user_check.data:
            supabase.table("user_cluesolver")\
                .insert({
                    "userid": user_id_value,
                    "cluesolveid": clue_id
                })\
                .execute()
        #Retrieving the answers and percentages given by the model
        stored_answers = supabase.table("answerssolved")\
            .select("*")\
            .eq("cluesolveid", clue_id)\
            .execute()
        if stored_answers.data:
            database_results = []
            database_row = stored_answers.data[0]
            #Returning the previous answers from the database instead of generating new ones
            for ans in range(1,11):
                answer = database_row.get(f"answer{ans}")
                confidence = database_row.get(f"perc{ans}")
                database_results.append({"answer": answer, "percentage": confidence})
            return JSONResponse({"results": database_results})
    #If this is a new clue, model will generate answers
    else:
        #Inserts the clue information into a new entry in the database
        insert_new_clue = supabase.table("cluesolver")\
            .insert({
                "clueentered": clue_text,
                "totalletters": answer_length,
                "lettersgiven": pattern
            })\
            .execute()
        clue_id = insert_new_clue.data[0]["cluesolveid"]
        #Retrieves the user ID of the player using their username
        user_id = supabase.table("users")\
            .select("userid")\
            .eq("username", clue_username)\
            .execute()
        user_id_value = user_id.data[0]["userid"]
        #Adding an entry showing this user has searched this clue
        supabase.table("user_cluesolver")\
            .insert({
                "userid": user_id_value,
                "cluesolveid": clue_id
            })\
            .execute()
        #Preparing to store the answers produced
        answers_record = {"cluesolveid": clue_id}
        #Prompt code continues as usual
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
            #Answers changed to max 20 characters, to fit the rules of the database
            t5_answers_ranked.append({"answer": answer[:20], "percentage": confidence, "pattern_match": pattern_match, "length_match": length_match})
        # Reverse = True is used to flip the list, as False == 0, comes before True == 1, when I actually need the True values to be first. Also need confidence levels to start with the highest, so descending order
        t5_answers_ordered = sorted(t5_answers_ranked, key = lambda ans: (ans["percentage"], ans["pattern_match"], ans["length_match"]), reverse = True)[:10]
        #Loops through each answer and adds them to the record
        for ans in range(len(t5_answers_ordered)):
            answers_record[f"answer{ans+1}"] = t5_answers_ordered[ans]["answer"]
            answers_record[f"perc{ans+1}"] = t5_answers_ordered[ans]["percentage"]
        #Inserts the record containing answers and percentages into the database
        supabase.table("answerssolved")\
            .insert(answers_record)\
            .execute()
        return JSONResponse({"results": t5_answers_ordered})
if __name__ == "__main__":
    uvicorn.run("T5 Model Backend:app", host="127.0.0.1", port=8000, reload=True)