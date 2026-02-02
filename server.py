# server.py - GitHub-safe version
import os
import sqlite3
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List
from groq import Groq
import google.generativeai as genai
import io
from dotenv import load_dotenv

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()  # loads .env file

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=GOOGLE_API_KEY)

# --- FASTAPI APP ---
app = FastAPI()

# --- DATABASE SETUP ---
db_connection = sqlite3.connect(":memory:", check_same_thread=False)
cursor = db_connection.cursor()
current_schema = ""
table_registry = []

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str

class ManualTableRequest(BaseModel):
    table_name: str
    csv_content: str

# --- HELPER FUNCTIONS ---
def get_table_schema(table_name, df):
    cols = []
    for col, dtype in df.dtypes.items():
        sql_type = "TEXT"
        if "int" in str(dtype):
            sql_type = "INTEGER"
        elif "float" in str(dtype):
            sql_type = "REAL"
        cols.append(f"{col} {sql_type}")
    return f"CREATE TABLE {table_name} ({', '.join(cols)});"

def execute_sql_query(sql):
    try:
        cursor.execute(sql)
        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            return {"columns": columns, "data": results, "error": None}
        else:
            return {"columns": [], "data": [], "error": "Query executed but returned no data."}
    except Exception as e:
        # Log internally, return generic error to user
        print(f"SQL Execution Error: {e}")
        return {"columns": [], "data": [], "error": "SQL execution failed. Check syntax or database."}

def generate_sql_gemini(system_prompt, user_question):
    model = genai.GenerativeModel('gemini-1.5-flash')
    full_prompt = f"{system_prompt}\n\nUser Question: {user_question}"
    response = model.generate_content(full_prompt)
    return response.text

def generate_sql_groq(system_prompt, user_question):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not found in environment variables")
    client = Groq(api_key=GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,
    )
    return chat_completion.choices[0].message.content

# --- ENDPOINTS ---
@app.get("/")
async def read_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Index file not found"}

@app.post("/reset")
async def reset_db():
    global db_connection, cursor, current_schema, table_registry
    db_connection.close()
    db_connection = sqlite3.connect(":memory:", check_same_thread=False)
    cursor = db_connection.cursor()
    current_schema = ""
    table_registry = []
    return {"message": "Database reset"}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    global current_schema
    uploaded_names = []

    for file in files:
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
            df.columns = [c.strip().replace(" ", "_") for c in df.columns]
            table_name = file.filename.replace(".csv", "").replace(" ", "_").lower()
            df.to_sql(table_name, db_connection, index=False, if_exists="replace")

            schema = get_table_schema(table_name, df)
            if table_name not in table_registry:
                current_schema += schema + "\n"
                table_registry.append(table_name)
            uploaded_names.append(table_name)
        except Exception as e:
            return JSONResponse(status_code=400, content={"message": f"Error processing file {file.filename}"})

    return {"message": "Success", "tables": table_registry}

@app.post("/create-table")
async def create_manual_table(request: ManualTableRequest):
    global current_schema
    try:
        data = io.StringIO(request.csv_content)
        df = pd.read_csv(data)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        table_name = request.table_name.replace(" ", "_").lower()
        df.to_sql(table_name, db_connection, index=False, if_exists="replace")

        schema = get_table_schema(table_name, df)
        if table_name not in table_registry:
            current_schema += schema + "\n"
            table_registry.append(table_name)

        return {"message": "Table created", "tables": table_registry}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to create table. Check CSV format.")

@app.post("/query")
async def process_query(request: QueryRequest):
    system_prompt = f"""
    You are an expert SQL assistant solving LeetCode SQL-style problems.
    
    SQLite database schema:
    {current_schema}
    
    CRITICAL RULES:
    1. Output only raw SQL, no markdown.
    2. Dialect: SQLite.
    3. Ranking & N-th Values: use DENSE_RANK() or LIMIT/OFFSET.
    4. Handle NULLs carefully; prefer NOT EXISTS over NOT IN (NULL).
    5. Prefer LEFT JOIN for optional matches.
    """

    generated_sql = ""
    used_model = "Gemini"

    try:
        generated_sql = generate_sql_gemini(system_prompt, request.question)
    except Exception as e_gemini:
        print(f"Gemini failed: {e_gemini}. Trying Groq...")
        used_model = "Groq"
        try:
            generated_sql = generate_sql_groq(system_prompt, request.question)
        except Exception as e_groq:
            raise HTTPException(
                status_code=500,
                detail="AI generation failed. Check API keys or question format."
            )

    # Clean response
    generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()

    result = execute_sql_query(generated_sql)
    return {"sql": generated_sql, "result": result, "model": used_model}

# --- MAIN ENTRY ---
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
