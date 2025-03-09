import cv2
import os
import pandas as pd
import numpy as np
from deepface import DeepFace
import time


# Configuration
REFERENCE_IMAGES_DIR = "./Faces"
REFERENCE_DATA_CSV = "citizens.csv"
SIMILARITY_THRESHOLD = 0.6  # Adjust as needed (0-1 range)
PROCESS_EVERY_N_FRAMES = 5  # Process every Nth frame for better performance
NEGATIVE_EMOTIONS = ['sad', 'angry', 'fear', 'disgust']  # List of negative emotions

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors using NumPy"""
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    
    # Avoid division by zero
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    
    similarity = dot_product / (magnitude_a * magnitude_b)
    return similarity


def create_alert(person_id):
    # Load CSV file
    df = pd.read_csv("citizens.csv")

    # Search for the person with matching ID
    person = df[df["ID"] == person_id]

    if person.empty:
        return "No record found for this ID."

    # Extract values
    name = person.iloc[0]["Legal Name"]
    occupation = person.iloc[0]["Occupation"]
    phone = person.iloc[0]["Phone Number"]
    emergency = person.iloc[0]["Emergency Contact"]
    sin = person.iloc[0]["Social Insurance Number"]
    address = person.iloc[0]["Address"]

    # Format message
    message = f"""
🚨 Possible Suicide Attempt Detected 🚨

👤 Name: {name}

💼 Occupation: {occupation}

📞 Phone Number: {phone}

📟 Emergency Contact: {emergency}

🆔 Social Insurance Number: {sin}

🏠 Home Address: {address}

⚠️ Please take immediate action
"""

    with open("./alert.txt", "w", encoding="utf-8") as file:
        file.write(message)

class FaceRecognitionSystem:
    def __init__(self):
        self.reference_data = {}  # HashMap for person data by ID
        self.reference_embeddings = []  # List of face embeddings
        self.reference_names = []  # List of names corresponding to embeddings
        
        # Load data
        self.load_reference_data()
        self.precompute_embeddings()
    
    def load_reference_data(self):
        """Load reference data from CSV into a hashmap"""
        # Create sample CSV if it doesn't exist
        if not os.path.exists(REFERENCE_DATA_CSV):
            print(f"Creating sample CSV file: {REFERENCE_DATA_CSV}")
            sample_data = pd.DataFrame([
                {'id': 'person1', 'name': 'John Doe', 'department': 'Engineering'},
                {'id': 'person2', 'name': 'Jane Smith', 'department': 'HR'}
            ])
            sample_data.to_csv(REFERENCE_DATA_CSV, index=False)
            print("Sample CSV created. Edit this file with your actual data.")
        
        # Load CSV
        if os.path.exists(REFERENCE_DATA_CSV):
            df = pd.read_csv(REFERENCE_DATA_CSV)
            print(f"Loaded CSV with {len(df)} entries")
            
            # Create hashmap for quick lookups by ID
            for _, row in df.iterrows():
                if 'id' in row:
                    self.reference_data[row['id']] = row.to_dict()
        else:
            print(f"Warning: Reference CSV not found at {REFERENCE_DATA_CSV}")
    
    def precompute_embeddings(self):
        """Precompute embeddings for all reference images (only done once)"""
        # Create directory if it doesn't exist
        if not os.path.exists(REFERENCE_IMAGES_DIR):
            os.makedirs(REFERENCE_IMAGES_DIR)
            print(f"Created reference images directory: {REFERENCE_IMAGES_DIR}")
            print("Place your reference face images in this folder.")
            return
        
        print("Precomputing face embeddings for all reference images...")
        start_time = time.time()
        
        # Process all images in the directory
        count = 0
        for filename in os.listdir(REFERENCE_IMAGES_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # Get person ID from filename
                    person_id = os.path.splitext(filename)[0]
                    
                    # Full path to the image
                    image_path = os.path.join(REFERENCE_IMAGES_DIR, filename)
                    
                    # Get this person's name from the CSV data (if available)
                    name = person_id
                    if person_id in self.reference_data:
                        name = self.reference_data[person_id].get('name', person_id)
                    
                    # Read the image directly with OpenCV
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Error: Could not read image {image_path}")
                        continue
                    
                    # Extract face embedding using DeepFace
                    embedding_objs = DeepFace.represent(
                        img_path=img,  # Pass the image array directly
                        model_name="VGG-Face",
                        enforce_detection=False,
                        detector_backend="opencv"
                    )
                    
                    # Extract the embedding vector from the first face
                    if embedding_objs and len(embedding_objs) > 0:
                        embedding = embedding_objs[0]["embedding"]
                        
                        # Store the embedding and name
                        self.reference_embeddings.append(embedding)
                        self.reference_names.append((person_id, name))
                        
                        count += 1
                        print(f"Processed {filename} ({name})")
                    else:
                        print(f"No face detected in {filename}, skipping")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        # Convert to numpy array for faster comparisons
        if count > 0:
            self.reference_embeddings = np.array(self.reference_embeddings)
            print(f"Reference embeddings shape: {self.reference_embeddings.shape}")
        
        # Report performance
        elapsed = time.time() - start_time
        print(f"Precomputed {count} face embeddings in {elapsed:.2f} seconds")
    
    def find_matching_face(self, face_img):
        """Find the matching face in the database of precomputed embeddings"""
        try:
            # Get embedding for current face
            embedding_objs = DeepFace.represent(
                img_path=face_img,  # Pass the face image directly
                model_name="VGG-Face", 
                enforce_detection=False,
                detector_backend="opencv"
            )
            
            # No face detected
            if not embedding_objs or len(embedding_objs) == 0:
                return False, None, None, 0
            
            # Get the embedding vector from the first face
            current_embedding = np.array(embedding_objs[0]["embedding"])
            
            # No reference faces to compare against
            if len(self.reference_embeddings) == 0:
                return False, None, None, 0
            
            # Calculate similarities for all reference embeddings
            similarities = []
            
            # Check if reference embeddings is a list of lists or a numpy array
            if isinstance(self.reference_embeddings, np.ndarray) and len(self.reference_embeddings.shape) == 2:
                # Calculate similarity with each reference embedding
                for ref_embedding in self.reference_embeddings:
                    similarity = cosine_similarity(current_embedding, ref_embedding)
                    similarities.append(similarity)
            else:
                # Fallback for non-numpy array case
                for ref_embedding in self.reference_embeddings:
                    similarity = cosine_similarity(current_embedding, ref_embedding)
                    similarities.append(similarity)
            
            # Find the best match
            similarities = np.array(similarities)
            max_similarity_idx = np.argmax(similarities)
            max_similarity = similarities[max_similarity_idx]
            
            # Check if the match is good enough
            if max_similarity > SIMILARITY_THRESHOLD:
                person_id, name = self.reference_names[max_similarity_idx]
                
                # Get all data for this person
                person_data = None
                if person_id in self.reference_data:
                    person_data = self.reference_data[person_id]
                else:
                    person_data = {'name': name}
                
                return True, person_id, person_data, max_similarity
            
            # No good match found
            return False, None, None, max_similarity
            
        except Exception as e:
            print(f"Error in face matching: {e}")
            return False, None, None, 0
    
    def run_webcam(self):
        """Run face recognition on webcam feed"""
        print("Starting webcam face recognition...")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        # Load face cascade for detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        print("Ready! Press 'q' to quit")
        
        # Skip frames counter for performance
        frame_count = 0
        
        # Track last recognized face to reduce flickering
        last_recognition = {
            'rect': None,
            'match_found': False,
            'person_id': None,
            'person_data': None,
            'similarity': 0,
            'emotion': None,
            'frames_ago': 999
        }
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            process_recognition = (frame_count % PROCESS_EVERY_N_FRAMES == 0)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Always detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            
            # If we found faces
            if len(faces) > 0:
                # Get the largest face
                largest_face = None
                largest_area = 0
                
                for (x, y, w, h) in faces:
                    area = w * h
                    if area > largest_area:
                        largest_area = area
                        largest_face = (x, y, w, h)
                
                x, y, w, h = largest_face
                
                # Always draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # Check if we should process recognition for this frame
                if process_recognition:
                    # Extract face region
                    face_roi = frame[y:y+h, x:x+w]

                    # Analyze emotion
                    emotion_analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    emotion = emotion_analysis[0]['dominant_emotion']

                    # Find matching face
                    match_found, person_id, person_data, similarity = self.find_matching_face(face_roi)

                    # Update last recognition
                    last_recognition = {
                        'rect': (x, y, w, h),
                        'match_found': match_found,
                        'person_id': person_id,
                        'person_data': person_data,
                        'similarity': similarity,
                        'emotion': emotion,
                        'frames_ago': 0
                    }

                    # Check if negative emotion is detected and we have a match
                    if match_found and emotion in NEGATIVE_EMOTIONS:

                        # Convert person_id to int and call get_person_info
                        person_id_int = int(person_id)
                        print(f"Negative emotion detected: {emotion}. Calling get_person_info for ID {person_id_int}")
                        create_alert(person_id_int)
                else:
                    # Increment age of last recognition
                    last_recognition['frames_ago'] += 1
                
                # Use last recognition data if it's recent enough
                if last_recognition['frames_ago'] < 30:  # About 1 second at 30fps
                    # Update rectangle color based on match status
                    if last_recognition['match_found']:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Display name and similarity
                        name = last_recognition['person_data'].get('name', last_recognition['person_id'])
                        display_text = f"{name} ({last_recognition['similarity']:.2f})"
                        cv2.putText(frame, display_text, 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # Display emotion
                        cv2.putText(frame, f"Emotion: {last_recognition['emotion']}", 
                                  (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Display additional data
                        y_offset = y+h+40
                        field_count = 0
                        for key, value in last_recognition['person_data'].items():
                            if key not in ['id', 'name'] and field_count < 3:
                                cv2.putText(frame, f"{key}: {value}", 
                                          (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                y_offset += 20
                                field_count += 1
                    else:
                        # Display "Unknown" and emotion
                        cv2.putText(frame, f"Unknown ({last_recognition['similarity']:.2f})", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        if last_recognition['emotion']:
                            cv2.putText(frame, f"Emotion: {last_recognition['emotion']}", 
                                      (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow("Face Recognition", frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run_webcam()