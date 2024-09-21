import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
# from flask_app_20240717_v100 import app
from claude_flask_consl_simple_webpage_v200_20240718 import app

if __name__ == "__main__":
    app.run()
