import json
from model import load_model
from query import process_query
from ultralytics import YOLO
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn


# Load the model registry
def load_model_registry():
    with open(r'D:\STUDENT-Final-2.v1i.yolov8\Models\model.json', 'r') as file:
        model_registry = json.load(file)
    return model_registry

def find_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return synonyms


def select_model(objects, model_registry):
    print(f"Extracted Objects from Query: {objects}")
    for model in model_registry["models"]:
        print(f"Model: {model['name']}, Classes: {model['classes']}")
        for obj in objects:
            synonyms = find_synonyms(obj)
            synonyms.add(obj) 
            if any(syn in class_name.replace('_', ' ') for syn in synonyms for class_name in model["classes"]):
                return model
    return None


#run_inference function using Ultralytics YOLO
def run_inference(model_path, image_path):
    model = YOLO(model_path) 
    results = model.predict(source=image_path, save=True) 
    print(type(results), results)
    return results


def main(query, image_path):
    objects = process_query(query)
    model_registry = load_model_registry()
    selected_model = select_model(objects, model_registry)
    
    if selected_model:
        print(f"Selected model: {selected_model['name']}")
        results = run_inference(selected_model["path"], image_path)
        
        if results and results[0]:
            # Assuming results is a Results object containing detected objects
            result = results[0]
            
            detected_objects = []
             # Convert detections to numpy array for easier processing
            detections = result.boxes.numpy()
            
            # Iterate over each detection in the boxes
            for det in detections:
                class_id = int(det[5])  # Access class_id correctly
                confidence = float(det[4])  # Access confidence score correctly
                label = results.names[class_id]  # Accessing class name using class ID
                
                if label.lower() in (obj.lower() for obj in objects):
                    detected_objects.append((label, confidence))
            
            if detected_objects:
                print(f"Detected objects related to the query '{query}':")
                for obj, score in detected_objects:
                    print(f"{obj} with confidence {score:.2f}")
            else:
                print("No relevant objects detected.")
        else:
            print("No objects detected or an error occurred.") # Handle the error case
    else:
        print("No suitable model found for the query.")




if __name__ == "__main__":
    user_query = "Detect students using mobile phones and looking around"
    image_path = r"D:\STUDENT-Final-2.v1i.yolov8\Test2\testimgs\img8.jpeg"
    main(user_query, image_path)