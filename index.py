import cv2
from deepface import DeepFace
import hashlib
import datetime
import json # For pretty printing block data

# --- Blockchain Simulation ---
class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data # Data can be age, emotion, etc.
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        """
        Calculates the hash of the block's content.
        """
        block_string = str(self.index) + \
                       str(self.timestamp) + \
                       json.dumps(self.data, sort_keys=True) + \
                       str(self.previous_hash)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def __repr__(self):
        return f"Block #{self.index}\nTimestamp: {self.timestamp}\nData: {json.dumps(self.data, indent=2)}\nHash: {self.hash}\nPrevious Hash: {self.previous_hash}\n"

# --- Blockchain Core Functions ---
def create_genesis_block():
    """
    Creates the first block in the blockchain (Genesis Block).
    """
    return Block(0, datetime.datetime.now(), {"genesis_block": True, "message": "First Block"}, "0")

def create_new_block(previous_block, data_for_block):
    """
    Creates a new block to add to the chain.
    """
    index = previous_block.index + 1
    timestamp = datetime.datetime.now()
    # data_for_block should be a dictionary e.g., {"age": 30, "emotion": "happy"}
    return Block(index, timestamp, data_for_block, previous_block.hash)

# Initialize the blockchain with the genesis block
blockchain = [create_genesis_block()]
previous_block = blockchain[0]
print("--- Blockchain Initialized ---")
print(previous_block)

# --- Webcam and Face Detection Setup ---
# Initialize webcam
cap = cv2.VideoCapture(0) # 0 is usually the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load face cascade for faster, initial face detection (optional, DeepFace can do it)
# Using OpenCV's Haar Cascade for a quick check if a face is present.
# DeepFace will then perform a more detailed analysis.
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise IOError("Failed to load Haar Cascade: haarcascade_frontalface_default.xml")
except Exception as e:
    print(f"Could not load face cascade classifier: {e}")
    print("Proceeding without pre-face detection using Haar Cascade. DeepFace will handle detection.")
    face_cascade = None


print("--- Starting Camera Feed and Analysis ---")
print("Press 'q' to quit.")

# Counter to limit how often DeepFace analysis is called (it's intensive)
frame_counter = 0
DEEPFACE_ANALYSIS_INTERVAL = 15 # Analyze every 15 frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # DeepFace expects RGB

    # Optional: Use Haar Cascade for initial, faster face detection
    faces_haar = []
    if face_cascade:
        faces_haar = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    frame_counter += 1

    # Perform DeepFace analysis periodically or if Haar cascade found faces
    # This is to manage performance as DeepFace analysis can be slow
    if frame_counter % DEEPFACE_ANALYSIS_INTERVAL == 0 or len(faces_haar) > 0:
        frame_counter = 0 # Reset counter
        try:
            # DeepFace.analyze can detect faces on its own if you don't pass face_detector_backend
            # It can analyze multiple faces in an image.
            # Actions: 'age', 'gender', 'emotion', 'race'
            # Enforce_detection=False will not raise an exception if no face is found.
            results = DeepFace.analyze(
                img_path=rgb_frame,
                actions=['age', 'emotion'],
                enforce_detection=False, # Don't error if no face
                silent=True # Suppress DeepFace's own console logs for cleaner output
            )

            # DeepFace returns a list of dictionaries, one for each detected face
            if isinstance(results, list):
                for result in results:
                    if result.get('face_confidence', 0) > 0.6: # Check if a face was confidently detected
                        age = result.get('age')
                        dominant_emotion = result.get('dominant_emotion')
                        region = result.get('region') # x, y, w, h

                        # Prepare data for the blockchain
                        data_to_store = {
                            "detected_at": datetime.datetime.now().isoformat(),
                            "age": age,
                            "emotion": dominant_emotion,
                            "face_region": region
                        }

                        # Create a new block and add it to our simulated blockchain
                        new_block = create_new_block(previous_block, data_to_store)
                        blockchain.append(new_block)
                        previous_block = new_block

                        print("\n--- New Block Added ---")
                        print(new_block)

                        # Draw rectangle and text on the frame
                        if region:
                            x, y, w, h = region['x'], region['y'], region['w'], region['h']
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            text = f"Age: {age}, Emotion: {dominant_emotion}"
                            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Handle cases where DeepFace might return a single dict (older versions or specific configs)
                # This part might need adjustment based on the exact DeepFace version and behavior
                if results and results.get('face_confidence', 0) > 0.6:
                    age = results.get('age')
                    dominant_emotion = results.get('dominant_emotion')
                    region = results.get('region')
                    data_to_store = {
                        "detected_at": datetime.datetime.now().isoformat(),
                        "age": age,
                        "emotion": dominant_emotion,
                        "face_region": region
                    }
                    new_block = create_new_block(previous_block, data_to_store)
                    blockchain.append(new_block)
                    previous_block = new_block
                    print("\n--- New Block Added (single result) ---")
                    print(new_block)
                    if region:
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        text = f"Age: {age}, Emotion: {dominant_emotion}"
                        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        except Exception as e:
            # This will catch errors from DeepFace.analyze if enforce_detection=True,
            # or other unexpected errors.
            # If enforce_detection=False, it usually returns an empty list or specific structure.
            # print(f"DeepFace analysis error or no face detected: {e}")
            pass # Silently pass if no face or minor issue, to keep the feed running

    # If not analyzing with DeepFace, still draw boxes from Haar if available
    # This provides faster visual feedback for face presence
    elif len(faces_haar) > 0 and not (frame_counter % DEEPFACE_ANALYSIS_INTERVAL == 0):
        for (x, y, w, h) in faces_haar:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)


    # Display the resulting frame
    cv2.imshow('Age and Emotion Detection - Blockchain Sim (Press Q to Quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("\n--- Program Terminated ---")
print(f"Total blocks in the chain: {len(blockchain)}")
cap.release()
cv2.destroyAllWindows()

# Optional: Print the entire blockchain at the end
# print("\n--- Final Blockchain State ---")
# for block in blockchain:
#     print(block)
#     print("-" * 20)

