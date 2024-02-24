import torch
import cv2
from ultralytics import YOLO
import nltk
from nltk.corpus import wordnet as wn
import operator
import json

# Load model registry (assuming it's populated with model names, paths, and classes)
def load_model_registry():
    with open(r'D:\STUDENT-Final-2.v1i.yolov8\Models\model.json', 'r') as file:
        model_registry = json.load(file)
    return model_registry

# Find synonyms for a word using WordNet
def find_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return synonyms

# Match user query to model classes using synonyms and fuzzy matching
def match_query_to_models(query, model_registry):
    matched_models = []
    query_words = set(query.lower().split())
    for model in model_registry["models"]:
        model_class_synonyms = {c.lower(): find_synonyms(c.lower()) for c in model["classes"]}
        matches = 0
        for word in query_words:
            for class_name, synonyms in model_class_synonyms.items():
                if word in synonyms or word == class_name.lower():
                    matches += 1
        if matches > 0:
            matched_models.append((model, matches))  # Store model and match count
    return sorted(matched_models, key=operator.itemgetter(1), reverse=True)  # Sort by match count

# Run inference on an image using the selected model
def run_inference(model_path, image_path):
    model = YOLO(model_path)
    results = model.predict(source=image_path, save=True)  # Perform object detection
    return results

# Extract relevant information from results based on user query and model classes
def extract_info(results, query, model_classes):
    detected_objects = []
    for det in results.boxes.numpy():
        class_id = int(det[5])
        confidence = float(det[4])
        label = results.names[class_id]
        if label.lower() in (word.lower() for word in query.lower().split()):
            detected_objects.append((label, confidence))
    info_text = ""  # Initialize empty info text
    for obj, score in detected_objects:
        if obj in model_classes:
            # Customize information extraction based on object and user query (replace with your logic)
            if obj == "student":
                info_text += f"Number of students: {len(detected_objects)}\n"
            elif obj == "phone":
                if "hand" in query.lower():
                    info_text += f"Students using phones: {len(detected_objects)}\n"
                else:
                    info_text += f"Phones detected: {len(detected_objects)}\n"
            else:
                info_text += f"Object: {obj}, Confidence: {score:.2f}\n"
    return info_text

# Main function for user interaction and inference
def main():
    model_registry = load_model_registry()
    while True:
        user_query = input("Enter your query (or 'q' to quit): ")
        if user_query.lower() == 'q':
            break

        image_path = input("Enter image path: ")

        # Match query to relevant models
        matched_models = match_query_to_models(user_query, model_registry)
        if not matched_models:
            print("No matching models found for your query.")
            continue

        # Select the best-matching model
        selected_model = matched_models[0][0]
        print(f"Selected model: {selected_model['name']}")

        # Run inference
        results = run_inference(selected_model["path"], image_path)

        # Extract relevant information
        model_classes = set(selected_model["classes"])
        info_text = extract_info(results, user_query, model_classes)

        # Provide output
        if info_text:
            print(f"\nInformation extracted from the image:\n{info_text}")
        else:
            print("No relevant information found in the image based on your query.")

if __name__ == "__main__":
    main()
