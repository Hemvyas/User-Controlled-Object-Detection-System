from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
from roboflow import Roboflow
import spacy
from spacy.matcher import PhraseMatcher

from main import (
    find_closest_standardized_class,
    classify_query
)  # Adapt the import statement based on where your functions are defined.
from config import API_KEY, API_URL

from flask_cors import CORS

app = Flask(__name__)
CORS(app,supports_credentials=True, resources={r"/upload": {"origins": "*"}})

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

STANDARDIZED_CLASSES = [
    "student",
    "chair",
    "table",
    "phone",
    "whiteboard",
    "id card",
    "hand raised",
    "lookaround",
    "up",
    "down",
    "notebook",
    "chair",
    "distracted",
    "attentive",
    "sleepy",
]

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

nlp = spacy.load("en_core_web_sm")

rf = Roboflow(API_KEY, API_URL)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # return jsonify({"message": "File received", "filename": file.filename})

    if file and allowed_file(file.filename):
        filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filename)

        # Process the image and get predictions
        user_query = request.form.get("query", "")  # Get the query from form data
        predictions = process_image_query(filename, user_query)

        # debug_info = process_image_query(filename, user_query)  # Assume this function is adjusted for debugging
        # return jsonify({"debug": debug_info, "message": "File received", "filename": file.filename})

        return jsonify({"predictions": predictions})
    else:
        return jsonify({"error": "File type not allowed"}), 400


def process_image_query(image_path, query):
    model_id, relevant_classes = classify_query(query)
    if not model_id:
        return {"error": "No suitable model found for your query."}

    try:
        model_endpoint, version = model_id.split("/")
        project = rf.workspace().project(model_endpoint)
        model = project.version(version).model

        # Perform inference
        results = model.predict(image_path, confidence=30, overlap=40).json()

        # Filter and refine predictions based on query
        for prediction in results["predictions"]:
            prediction["class"] = find_closest_standardized_class(
                prediction["class"], STANDARDIZED_CLASSES
            )

        if relevant_classes:
            filtered_predictions = [
                prediction
                for prediction in results["predictions"]
                if prediction["class"] in relevant_classes
            ]
        else:
            filtered_predictions = results["predictions"]

        # Return filtered predictions
        return {"predictions": filtered_predictions}

    except Exception as e:
        return {"error": str(e)}


@app.route("/files/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
