# import cv2
# from roboflow import Roboflow
# import os
# from config import API_KEY, API_URL, QUERY_TO_MODEL_ID
# import spacy

# nlp = spacy.load("en_core_web_sm")


# def classify_query(query):
#     """
#     Classifies the query to find the most relevant model.
#     Uses Spacy for NLP to enhance query understanding.
#     """
#     doc = nlp(query.lower())
#     max_score = 0
#     selected_model = None

#     for model_query, model_id in QUERY_TO_MODEL_ID.items():
#         score = 0
#         model_keywords = set(model_query.split())
#         query_keywords = set(
#             [token.lemma_ for token in doc if token.pos_ in ("NOUN", "VERB", "ADJ")]
#         )

#         common_elements = model_keywords.intersection(query_keywords)
#         score = len(common_elements)

#         if score > max_score:
#             max_score = score
#             selected_model = model_id

#     if selected_model is None:
#         # Fallback or default model
#         selected_model = "default-model-id/1"  # Example, replace with an actual model ID if you have a general-purpose fallback model.

#     return selected_model


# rf = Roboflow(API_KEY, API_URL)

# if __name__ == "__main__":
#     user_query = input("Enter your query (e.g., 'count students', 'find phones'): ")

#     model_details = classify_query(user_query)
#     if not model_details:
#         print("Invalid query or no suitable model found. Please try again.")
#         exit()

#     model_endpoint, version = model_details.split("/")
#     project = rf.workspace().project(model_endpoint)
#     model = project.version(version).model

#     # Assuming "your_image.jpg" is the path to the image you want to analyze
#     image_path = r"D:\User-Controlled-Object-Detection-System\Test1\testimgs\img2.jpg"  # Update this path
#     results = model.predict(image_path, confidence=50, overlap=30).json()

#     # Process and display results
#     print(results)


import cv2
from roboflow import Roboflow
import os
import spacy
from spacy.matcher import PhraseMatcher

# Load the spaCy language model (replace with a different model if needed)
nlp = spacy.load("en_core_web_lg")
from config import API_KEY, API_URL, QUERY_TO_MODEL_ID

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


def refine_relevant_classes(query, possible_classes):
    doc = nlp(query.lower())
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(text) for text in possible_classes]
    matcher.add("Classes", patterns)

    matches = matcher(doc)
    refined_classes = set()
    for match_id, start, end in matches:
        refined_classes.add(doc[start:end].text)

    return list(refined_classes)


def find_closest_standardized_class(detected_class, standardized_classes):
    detected_class_doc = nlp(detected_class)
    max_similarity = 0.0
    closest_class = detected_class
    for std_class in standardized_classes:
        std_class_doc = nlp(std_class)
        similarity = detected_class_doc.similarity(std_class_doc)
        if similarity > max_similarity:
            max_similarity = similarity
            closest_class = std_class
    return closest_class


def classify_query(query):
    doc = nlp(query.lower())
    max_score = 0
    selected_model = None
    relevant_classes = []

    for model_query, (model_id,classes) in QUERY_TO_MODEL_ID.items():
        score = 0
        model_keywords = set(model_query.split())
        query_keywords = set(
            [token.lemma_ for token in doc if token.pos_ in ("NOUN", "VERB", "ADJ")]
        )

        # Calculate the Jaccard similarity score for more robust matching
        common_elements = len(model_keywords.intersection(query_keywords))
        union_elements = len(model_keywords.union(query_keywords))
        if union_elements > 0:
            score = common_elements / union_elements
        print(f"Model Query: {model_query}, Score: {score}")  # Debugging

        if score > max_score:
            max_score = score
            selected_model = model_id
            relevant_classes = classes
            print(relevant_classes)
    if not selected_model:
        print("No suitable model found for your query. Using the default model.")
        selected_model = (
            "crowd-count-rcr1e/1"  # Replace with your actual fallback model ID
        )
        relevant_classes = []
    # Convert relevant_classes to standardized classes for consistency
    relevant_classes = [
        find_closest_standardized_class(cls, STANDARDIZED_CLASSES)
        for cls in relevant_classes
    ]
    return selected_model, relevant_classes


def process_image(image_path, user_query):
    model_id, relevant_classes = classify_query(user_query)
    if not model_id:
        return "No suitable model found for your query."

    try:
        rf = Roboflow(API_KEY, API_URL)

        model_endpoint, version = model_id.split("/")
        project = rf.workspace().project(model_endpoint)
        model = project.version(version).model

        results = model.predict(image_path, confidence=30, overlap=40).json()

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
            results["predictions"] = filtered_predictions

        return results
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    user_query = input("Enter your query (e.g., 'count students', 'find phones'): ")
    image_path = "path/to/your/test/image.jpg"  # Update this path for testing
    results = process_image(image_path, user_query)
    print(results)
