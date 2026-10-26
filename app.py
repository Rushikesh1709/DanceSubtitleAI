"""Run the web application:  python app.py"""
from app import create_app
from config import Config

app = create_app()

if __name__ == "__main__":
    print(f"\n  DanceSubtitleAI is running.  Open  http://{Config.HOST}:{Config.PORT}  in your browser.\n"
          "  Press CTRL+C in this window to stop it.\n")
    # threaded=True: the page stays usable while a video is being analysed in the background.
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG, threaded=True, use_reloader=False)
