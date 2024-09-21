import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
# from flask_app_20240717_v100 import app
from chatgpt_flask_gunicorn_nginx_wsgi_streaming_v100_20240718 import app

if __name__ == "__main__":
    app.run()
