from flask import Flask

app = Flask(__name__)


@app.route('/')
def init():
    return 'Chat Server On'


if __name__ == "__main__":
    app.run(port=5000)