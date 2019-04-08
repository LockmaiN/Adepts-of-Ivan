import random
import cv2
import numpy as np
from flask import Flask, render_template, flash, redirect, session, request, Response
from collections import deque
from wtforms import Form
from flask_wtf.file import FileField
from tempfile import NamedTemporaryFile
from configparser import ConfigParser
from pathlib import Path
from functools import wraps


app = Flask(__name__)
vc = cv2.VideoCapture('/home/dsabodashko/Downloads/cam4.render.mp4')
config_path = str(Path(__file__).parent / 'config.conf')
config = ConfigParser()
config.read(config_path)

emotions = deque(maxlen=60)


def login_required(func):

    @wraps(func)
    def login_helper(*args, **kwargs):
        if session.get('logged_in'):
            return func(*args, **kwargs)
        else:
            return Response(status=401)
    return login_helper


@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect('login')
    else:
        return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    data = {
        'message': ""
    }
    if request.method == 'GET':
        return render_template('login.html')
    if request.method == 'POST':
        if request.form['password'] == 'admin' and request.form['username'] == 'admin':
            session['logged_in'] = True
            return redirect('/')
        else:
            data['message'] = "login or password is incorrect"
            return render_template('login.html', data=data)


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session['logged_in'] = False
    return redirect('login')


def gen_frame():
    """Video streaming generator function."""
    while True:
        rval, frame = vc.read()
        tmp_file = NamedTemporaryFile(suffix='.jpg')
        cv2.imwrite(tmp_file.name, frame)
        tmp_file.seek(0)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open(tmp_file.name, 'rb').read() + b'\r\n')


# def gen_stats():
#     """Video streaming generator function."""
#     while True:
#
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + open(tmp_file.name, 'rb').read() + b'\r\n')


@app.route('/render-video', methods=['GET', 'POST'])
@login_required
def render_video():
    # TODO implement emotion recognition from video file
    return render_template('render_video.html')


@app.route('/video_stream')
def video_stream():
    return render_template('video_stream.html')


@app.route('/video_feed')
@login_required
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/about')
@login_required
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.secret_key = 'secret_key_1234345'
    app.run(debug=True, host=config.get('service', 'host'), port=int(config.get('service', 'port')))
