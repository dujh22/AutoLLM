from flask import Flask
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/video_feed')
def video_feed():
    logging.info("Received request for /video_feed")
    return 'Hello World'

if __name__ == "__main__":
    logging.info("Starting Flask server on port 8010...")
    app.run(host='0.0.0.0', port=8010, threaded=True)

