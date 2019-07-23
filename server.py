# -*- coding: utf-8 -*-
import os
from flask import *

app = Flask(__name__)
app._static_folder = "static"

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

@app.route("/", methods=["GET", "POST"])
def main_page():
    return render_template("index.html")#, title="Result edition")

if __name__ == '__main__':
    app.debug = True
    app.run()#host='0.0.0.0', port=80, threaded=True)
