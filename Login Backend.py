#Login Backend File
#Imports flask to connect to database
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
#Imports os and dotenv to connect securely
import os
from dotenv import load_dotenv
app = Flask(__name__)
#Policies to let JSON through port
CORS(app,resources = {r"/*": {"origins": "http://127.0.0.1:5500"}})
load_dotenv()
supabase_url = os.getenv("supabase_url")
supabase_service_key = os.getenv("supabase_service_key")
supabase: Client = create_client(supabase_url, supabase_service_key)
#Frontend received at this directory
@app.post("/userLogin")
#Function to check database and login
def login():
    user_info = request.get_json()
    username = user_info.get("username")
    password = user_info.get("password")
    #Searching database for matching username and password
    search_result = supabase.table("users")\
        .select("*")\
        .eq("username", username)\
        .eq("password", password)\
        .execute()
    user_data = search_result.data
    #If successful, user can be logged in
    if len(user_data) == 1:
        return jsonify({"success": True, "username": username})
    else:
        return jsonify({"success": False})
if __name__ == "__main__":
    app.run(debug = True)