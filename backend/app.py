from flask import Flask
from flask_cors import CORS
from api.routes import api

# ✅ Create app FIRST
app = Flask(__name__)

# ✅ Enable CORS
CORS(app)

# ✅ Register blueprint AFTER app is created
app.register_blueprint(api, url_prefix="/api")

# ✅ Root route
@app.route('/')
def home():
    return {"message": "Stock AI Backend Running 🚀"}

# ✅ Run server
if __name__ == '__main__':
    app.run(debug=True)