from flask import Flask
from filters import filters_bp
from background import bg_bp
from visualizer import vis_bp

app = Flask(__name__)

# Register the separate modules
app.register_blueprint(filters_bp)
app.register_blueprint(bg_bp)
app.register_blueprint(vis_bp)

if __name__ == '__main__':
    print("🚀 Starting Modular CamStudio Pro...")
    app.run(debug=True, threaded=True)