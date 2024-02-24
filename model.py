import json
from nltk.corpus import wordnet as wn 

def load_model_registry():
    with open(r'D:\STUDENT-Final-2.v1i.yolov8\Models\model.json', "r") as f:
        return json.load(f)

def find_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return synonyms

def select_model(objects, model_registry):
    for model in model_registry["models"]:
        class_names = [cls.lower() for cls in model["classes"]]  # Convert to lowercase
        for obj in objects:
            synonyms = find_synonyms(obj)
            synonyms.add(obj)
            if any(syn in class_name for syn in synonyms for class_name in class_names):
                return model
    return None


# def process_image(image_data, query):
#     image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

#     # Extract relevant noun objects from query
#     doc = nlp(query)
#     objects = [token.text.lower() for token in doc if token.pos_ == "NOUN"]

#     selected_model_info = select_model(objects, app.model_registry)
#     detected_objects = []  # Initialize detected_objects here

#     if selected_model_info:
#         model = YOLO(selected_model_info['path'])
#         results = model(image)  # Ensure 'results' is set here
#         print(f"Model inference output: {results}")

#         if hasattr(results, 'pred') and len(results.pred):
#             detections = results.pred[0]
#             info_text = ""

#             for det in detections:
#                 xmin, ymin, xmax, ymax, confidence, class_id = det[:6]
#                 label = results.names[int(class_id)]
#                 if label.lower() in objects:
#                     detected_objects.append((label, confidence.item()))
#                     info_text += f"Object: {label}, Confidence: {confidence:.2f}\n"
                    
#             _, buffer = cv2.imencode('.jpg', image)
#             preview_image = buffer.tobytes()
#             encoded_image = base64.b64encode(preview_image).decode('utf-8')
#             return info_text, encoded_image, detected_objects  # You might also want to return detected_objects depending on your needs
#         else:
#             return "Error: Model did not return expected results format.", None, []
#     else:
#         return "No suitable model found for the given query.", None, []