# app.py
from flask import Flask, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from flask_caching import Cache
import os

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='.')
CORS(app)

# Initialize cache
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)

# Get the API Key
app.config['GOOGLE_MAPS_API_KEY'] = os.getenv('GOOGLE_MAPS_API_KEY')

# Register the blueprint (import after cache is initialized)
from api import api_blueprint
app.register_blueprint(api_blueprint)

# Serve the frontend
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    app.run(debug=True)
