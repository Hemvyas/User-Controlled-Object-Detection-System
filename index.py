import json
import numpy as np
import cv2
from flask import Flask, request, jsonify
from ultralytics import YOLO
import spacy
from nltk.corpus import wordnet as wn
from flask_cors import CORS
import base64


def levenshtein_distance(a, b):
    """Calculates the Levenshtein distance between two strings."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    d = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        d[i, 0] = i
    for j in range(m + 1):
        d[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return d[n, m]


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


nlp = spacy.load("en_core_web_sm")


def load_model_registry():
    with open(
        r"D:\User-Controlled-Object-Detection-System\Models\model.json", "r"
    ) as f:
        return json.load(f)


app.model_registry = load_model_registry()  


def find_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))

    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            for other_syn in wn.synsets(lemma.name().replace("_", " ")):
                similarity = syn.wup_similarity(other_syn)
                if similarity > 0.8:  
                    synonyms.add(other_syn.lemma_names()[0].replace("_", " "))
    return synonyms


def select_model(objects, model_registry):
    best_model = None
    highest_score = -1

    for model in model_registry["models"]:
        score = 0
        class_names = [cls.lower() for cls in model["classes"]]

        for obj in objects:
            synonyms = find_synonyms(obj)
            synonyms.add(obj)

            score += sum(1 for syn in synonyms if syn in class_names)

            for cls in class_names:
                distance = levenshtein_distance(obj, cls)
                if distance <= 2: 
                    score += 0.5  

            overlap_count = sum(1 for cls in class_names if any(syn in cls for syn in synonyms))
            score += overlap_count * 0.2

        if score > highest_score:
            highest_score = score
            best_model = model

    return best_model


@app.route("/process", methods=["POST"])
def process_input():
    input_type = request.form["inputType"]
    inputData = request.files["inputData"].read() if input_type == "image" else None
    query = request.form["query"]
    # print(f"Received query: {query}")  # Add this line to log the query

    if input_type == "image":
        info_text, preview, detected_objects = process_image(inputData, query)
        return jsonify(
            {"infoText": info_text, "preview": preview, "detections": detected_objects}
        )  
    else:
        return jsonify({"error": "Invalid input type"}), 400


def process_image(image_data, query):
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    doc = nlp(query)

    objects = set()
    for chunk in doc.noun_chunks:  
        objects.add(chunk.text.lower())
    for ent in doc.ents:  
        objects.add(ent.text.lower())

    objects = list(objects)

    selected_model_info = select_model(objects, app.model_registry)
    detected_objects = []  

    if selected_model_info:
        model = YOLO(selected_model_info["path"])
        results = model(image)  

        results = model(image)
        print("Raw model output:", results)

        if hasattr(results, "xyxy") and len(results.xyxy):
            print("xyxy attribute exists")
            detections = results.xyxy[0]
            info_text = ""

            for det in detections:
                print("inside")
                xmin, ymin, xmax, ymax, confidence, class_id = det[:6].cpu().numpy()
                label = results.names[int(class_id)]
                print(f"Detection: {label}, Confidence: {det[4]}")
                if label.lower() in objects:
                    print("inside ejdh")
                    # detected_objects.append((label, confidence.item()))
                    detected_objects.append(
                        {
                            "label": label,
                            "confidence": round(confidence.item(), 2),
                            "bounding_box": {
                                "xmin": int(xmin),
                                "ymin": int(ymin),
                                "xmax": int(xmax),
                                "ymax": int(ymax),
                            },
                        }
                    )
                info_text += f"Object: {label}, Confidence: {confidence:.2f}, Box: [{xmin}, {ymin}, {xmax}, {ymax}]\n"
                # print(info_text,confidence,label)
                _, buffer = cv2.imencode(".jpg", image)
                # preview_image = buffer.tobytes()
                preview_image = base64.b64encode(preview_image).decode("utf-8")
                return info_text, preview_image, detected_objects

            else:
                return "Error: Model did not return expected results format.", None, []
        else:
            return "No suitable model found for the given query.", None, []


if __name__ == "__main__":
    app.run(debug=True)
