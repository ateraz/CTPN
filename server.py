import os
import subprocess

from flask import Flask, request, abort
from werkzeug.utils import secure_filename

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif"])
SCRIPT = "python tools/demo.py --no-gpu"
app.config["UPLOAD_FOLDER"] = "./uploads"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        abort(401)

    file = request.files["file"]
    if not allowed_file(file.filename):
        abort(401)

    file_name = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file_name)
    file.save(file_path)
    process = subprocess.Popen(SCRIPT.split(), stdout=subprocess.PIPE)
    os.remove(file_path)
    out, _ = process.communicate()
    return out

app.run(debug=True)
