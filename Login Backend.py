from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
app = Flask(__name__)
CORS(app,resources = {r"/*": {"origins": "http://127.0.0.1:5500"}})
supabase_url = "https://zduutknmvdwdubjcsmao.supabase.co"
supabase_service_key = "sb_secret_tNKw9op9nq6WRyWpv9QsqQ_6eA2mnaT"
supabase: Client = create_client(supabase_url, supabase_service_key)
@app.post("/userLogin")
def login():
    user_info = request.get_json()
    username = user_info.get("username")
    password = user_info.get("password")
    search_result = supabase.table("users")\
        .select("*")\
        .eq("username", username)\
        .eq("password", password)\
        .execute()
    user_data = search_result.data
    if len(user_data) == 1:
        return jsonify({"success": True, "username": username})
    else:
        return jsonify({"success": False})
if __name__ == "__main__":
    app.run(debug = True)