
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.get("/")
def hello():
    return "DFText server is up!"

# 🔹 Add your routes below

# no app.run()
