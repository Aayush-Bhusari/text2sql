# text2sql
Simple natural language to SQL queries
<img width="1918" height="836" alt="image" src="https://github.com/user-attachments/assets/a3236e73-697b-424b-a464-879c8e3fb242" />


Features:
Upload CSVs or create tables manually
Ask questions in plain English
View generated SQL and query results
AI fallback: Gemini → Groq
Tech: FastAPI, SQLite, Python, Vanilla JS, Google Gemini & Groq APIs

Setup:
Install dependencies: pip install -r requirements.txt
Create .env with your API keys:

GOOGLE_API_KEY=your_google_key
GROQ_API_KEY=your_groq_key
HOST=
PORT=

Run: python server.py
