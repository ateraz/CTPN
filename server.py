import os
import subprocess

from flask import Flask, request, abort, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='web', static_folder='web')

ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif"])
SCRIPT = "python2.7 tools/demo.py --no-gpu" 
app.config["UPLOAD_FOLDER"] = "./uploads"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/result", methods=["POST"])
def upload_file():
    print request.files
    if "file" not in request.files:
        abort(400)

    file = request.files["file"]
    if not allowed_file(file.filename):
        abort(400)

    file_name = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file_name)
    file.save(file_path)
    process = subprocess.Popen(SCRIPT.split(), stdout=subprocess.PIPE)
    out, _ = process.communicate()
    os.remove(file_path)
    return jsonify({
        'text': out,
        'image': 'processed/' + file_name.replace('.', '_with_boxes.')
    })

@app.route("/")
def index():
    return render_template('index.html')

app.run(debug=True, host='0.0.0.0')
