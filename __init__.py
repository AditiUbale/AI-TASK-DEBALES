from flask import Flask

app = Flask(__name__)

# Import routes from chatbot.py
from app import chatbot
