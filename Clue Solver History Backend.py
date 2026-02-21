#Importing Flask to access the supabase database
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
import os
from dotenv import load_dotenv
app = Flask(__name__)
#Using CORS policies to make sure the port is allowed
CORS(app,resources = {r"/*": {"origins": "http://127.0.0.1:5500"}})
#Connecting to the database securely
load_dotenv()
supabase_url = os.getenv("supabase_url")
supabase_service_key = os.getenv("supabase_service_key")
supabase: Client = create_client(supabase_url, supabase_service_key)
#Sending the JSON to this directory, for the frontend to be able to access it
@app.post("/clueSolverHistory")
#Function to search the database
def displayHistory():
    username_info = request.get_json()
    username = username_info.get("username")
    #Finding the player's user ID in the database
    user_id = supabase.table("users")\
            .select("userid")\
            .eq("username", username)\
            .execute()
    user_id_value = user_id.data[0]["userid"]
    #Searching for clues that the player has solved before
    user_check = supabase.table("user_cluesolver")\
            .select("cluesolveid")\
            .eq("userid", user_id_value)\
            .execute()
    #If none are found, false is returned
    if not user_check.data:
        return jsonify({"success": False})
    clue_solver_ids = []
    #Stores all instances of clue IDs, as user is likely to have solved multiple clues before
    for row in user_check.data:
        clue_solver_ids.append(row["cluesolveid"])
    clue_history = []
    for clue_id in clue_solver_ids:
        #Retrieving the information they entered for each clue
        clue_information = supabase.table("cluesolver")\
            .select("clueentered", "totalletters", "lettersgiven")\
            .eq("cluesolveid", clue_id)\
            .execute()
        if not clue_information.data:
            continue
        clue_info = clue_information.data[0]
        #Retrieving the answers and percentages the model provided for each clue
        answer_information = supabase.table("answerssolved")\
            .select("*")\
            .eq("cluesolveid", clue_id)\
            .execute()
        if not answer_information.data:
                continue
        answer_info = answer_information.data[0]
        #Storing all the information in one place
        full_clue = {
                "clue": clue_info["clueentered"],
                "length": clue_info["totalletters"],
                "pattern": clue_info["lettersgiven"],
                "answers": []
        }
        #Loops through each answer to store it
        for ans in range(1,11):
            answer = f"answer{ans}"
            confidence = f"perc{ans}"
            #Appending the answers and percentages
            full_clue["answers"].append({
                "answer": answer_info[answer],
                "confidence": answer_info[confidence]
            })
        clue_history.append(full_clue)
    #Returning all clues and their information
    return jsonify({"success": True, "history": clue_history})
#Runs the backend
if __name__ == "__main__":
    app.run(debug = True, port = 5001)
