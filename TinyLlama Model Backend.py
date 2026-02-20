#Importing libraries to access the model
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
#Tokenizing and loading the merged model
model_location = "./clueGeneratorMerged"
tokenizer = AutoTokenizer.from_pretrained(model_location)
llama_model = AutoModelForCausalLM.from_pretrained(
    model_location,
    device_map = "mps",
    dtype = torch.float16
)
llama_model.eval()
#Function to generate answers that fit the theme
def generate_answers(theme):
    #Engineered prompt to give answers fitting to the theme - model has a pattern to follow which helps keep it on track
    llama_prompt = f"""
Continue this chain:
Theme: Animals
Answers:
CAT,
DOG,
HORSE,
SHEEP,
ZEBRA

Theme: Fruit
Answers:
APPLE,
GRAPE,
STRAWBERRY,
PEAR,
MANGO

Theme: Flowers
Answers:
ROSE,
DAISY,
TULIP,
BLUEBELL,
LILY

Theme: {theme}
Answers:

    """
    llama_inputs = tokenizer(llama_prompt, return_tensors = "pt").to("mps")
    #Parameters for generating answers, tailored to give the best responses possible
    with torch.no_grad():
        llama_outputs = llama_model.generate(
            **llama_inputs,
            max_new_tokens = 200,
            temperature = 0.85,
            top_p = 0.9,
            repetition_penalty = 1.15,
            do_sample = True,
            pad_token_id = tokenizer.eos_token_id
        )
    #Decoding the results given
    llama_results = tokenizer.decode(llama_outputs[0], skip_special_tokens = True)
    #Sometimes the original prompt is copied in at the top, so am removing this
    llama_results_formatted = llama_results[len(llama_prompt):].strip()
    #Extra characters are removed, keeping only the valid words
    candidate_answers = re.findall(r'[A-Za-z]+(?:\s[A-Za-z]+)*', llama_results_formatted)
    valid_answers_unique = []
    #Joining up multiple word answers and giving the length of each word
    for answer in candidate_answers:
        words = answer.split()
        valid_answer = ''.join(words).upper()
        #No duplicates, and length of answer must be reasonable
        if valid_answer not in valid_answers_unique and 3 <= len(valid_answer) <= 15:
            answer_length = ",".join(str(len(word))for word in words)
            valid_answers_unique.append((valid_answer, answer_length))
    #Returns the top 10 answers - can be amended later to fit requirements
    return valid_answers_unique[:10]
#Function to generate clues based off of the answers given
def generate_clues(answer, theme):
    #Engineered prompt does not really work as desired - is very hit and miss, but unfortunately hardware and time limitations means I cannot improve it any more
    llama_prompt = f"""
You are a professional crossword setter.

Write a concise crossword clue for {answer}.

Theme: {theme}
Answer: {answer}

Rules:
- One short clue only
- Do NOT include the answer in the clue
- Do NOT include explanations or sources
- No extra text

Clue:
"""
    llama_inputs = tokenizer(llama_prompt, return_tensors = "pt").to("mps")
    #Using the same parameters as before, to ensure consistency
    with torch.no_grad():
        llama_outputs = llama_model.generate(
            **llama_inputs,
            max_new_tokens = 200,
            temperature = 0.85,
            top_p = 0.9,
            repetition_penalty = 1.15,
            do_sample = True,
            pad_token_id = tokenizer.eos_token_id
        )
    #Decoding the clue for each answer
    llama_results = tokenizer.decode(llama_outputs[0], skip_special_tokens = True)
    llama_results_formatted = llama_results[len(llama_prompt):].strip()
    return llama_results_formatted
#Runs the functions with the theme inputted
if __name__ == "__main__":
    theme = input("Enter a theme: ")
    answers = generate_answers(theme)
    print(answers)
    #Gives a clue for each answer
    for answer in answers:
        clue = generate_clues(answer, theme)
        print(clue)
