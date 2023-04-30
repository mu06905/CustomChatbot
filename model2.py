import openai
import os
import numpy as np
import json
from flask import Flask, render_template, request
import sqlite3

# Set up OpenAI API credentials
openai.api_key = "add here"

# Set up Flask app
app = Flask(__name__)

# Define function to create embeddings for a file
def create_embeddings(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        text = f.read()
        embedding = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(embedding['data'][0]['embedding'])

# Define function to search for closest matching file
def search_files(query_embedding, file_embeddings):
    # similarity_scores = np.dot(file_embeddings.T, query_embedding) / (np.linalg.norm(file_embeddings, axis=1) * np.linalg.norm(query_embedding))

    similarity_scores = np.dot(file_embeddings, query_embedding) / (np.linalg.norm(file_embeddings) * np.linalg.norm(query_embedding))
    best_match_index = np.argmax(similarity_scores)
    return best_match_index

# Define function to load file embeddings from a directory and store them in a database
def load_file_embeddings(directory_path, db_path):
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, embedding BLOB)")
        
        # Get the list of filenames already present in the table
        c.execute("SELECT filename FROM files")
        existing_filenames = set(row[0] for row in c.fetchall())

        # Insert embeddings for files that are not already present
        for filename in os.listdir(directory_path):
            if filename not in existing_filenames:
                file_path = os.path.join(directory_path, filename)
                file_embedding = create_embeddings(file_path)
                c.execute("INSERT INTO files (filename, embedding) VALUES (?, ?)", (filename, file_embedding.tobytes()))

        conn.commit()


# Load file embeddings from a directory and store them in a database
directory_path = "data"
db_path = "file_embeddings.db"
load_file_embeddings(directory_path, db_path)

# Define function to generate answer
def generate_answer(question, directory_path, db_path):
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM files")
        rows = c.fetchall()
        if not rows:
            return "I couldn't find the answer to that question in your files."
        else:
            file_embeddings = []
            filenames = []
            print(len(rows))
            for row in rows:
                filenames.append(row[1])
                file_embeddings.append(np.frombuffer(row[2])) 
            query_embedding = np.array(openai.Embedding.create(
                model="text-embedding-ada-002", 
                input=question
            )['data'][0]['embedding'])
            best_match_index = search_files(query_embedding, file_embeddings)
            best_match_filename = filenames[best_match_index]
            with open(os.path.join(directory_path, best_match_filename), 'r', encoding="utf-8") as f:
                file_text = f.read()
            answer = openai.Completion.create(
                model="text-davinci-002",
                prompt="Given a question, try to answer it using the content of the file extracts below, and if you cannot answer, or find a relevant file, just output \"I couldn't find the answer to that question in your files.\".\n\n" +
                       "If the answer is not contained in the files or if there are no file extracts, respond with \"I couldn't find the answer to that question in your files.\" If the question is not actually a question, respond with \"That's not a valid question.\"\n\n" +
                       file_text + "\n\nQuestion: " + question + "\nAnswer:",
                temperature=0.6,
                max_tokens=512,
                n = 1,
                stop=None,
                timeout=10,
                frequency_penalty=0,
                presence_penalty=0
            )
            if answer.choices[0].text == "I couldn't find the answer to that question in your files.":
                return answer.choices[0].text
            else:
                return answer.choices[0].text.strip()


messages = []
@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        question = request.form['question']
        answer = generate_answer(question, directory_path, db_path)
        messages.append({'type': 'question', 'text': question})
        messages.append({'type': 'answer', 'text': answer})
   
    return render_template('chat.html', messages=messages)


if __name__ == '__main__':
    app.run(debug=True)
