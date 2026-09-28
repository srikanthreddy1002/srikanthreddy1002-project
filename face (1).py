import cv2
import face_recognition
import os
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Folder to store known faces
KNOWN_FACES_DIR = "known"
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Email configuration
EMAIL_USER = "angothumuni1729@gmail.com"
EMAIL_PASS = "rxhfuepymdxvrfls"
VAPID_MAILTO = "nenavathdevaraj750@gmail.com"

# Authorized people whitelist
authorized_people = ["puneeth", "srikanth"]

def send_email_notification(person_name):
    """Send email notification when a face is recognized."""
    try:
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = VAPID_MAILTO
        msg['Subject'] = f"Face Recognized: {person_name}"
        
        body = f"A face has been recognized: {person_name}\nTime: {np.datetime64('now')}"
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        print(f"Email sent successfully for: {person_name}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0].lower())
            print(f"Loaded {filename}")

cap = cv2.VideoCapture(0)
mode = "idle"

print("Press 's' to save face, 'r' to recognize, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    key = cv2.waitKey(1) & 0xFF

    # 🔹 Save face
    if key == ord('s'):
        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:
            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            
            name = input("Enter the name of the person: ").strip().lower()
            if name == "":
                name = "unknown"
            
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)
            
            top, right, bottom, left = face_locations[0]
            face_image = frame[top:bottom, left:right]
            cv2.imwrite(f"{KNOWN_FACES_DIR}/{name}.jpg", face_image)
            print(f"Face saved for {name}!")

    # 🔹 Start recognition
    elif key == ord('r'):
        mode = "recognize"
        print("Recognition mode ON")

    elif key == ord('q'):
        break

    # 🔹 Recognition mode
    if mode == "recognize" and known_face_encodings:
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            box_color = (0, 0, 255)

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    # Send email when face is recognized
                    send_email_notification(name)
                    
                    if name in authorized_people:
                        box_color = (0, 255, 0)
                    else:
                        box_color = (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.putText(frame, name, (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

    cv2.imshow("Face Recognition", frame)

cap.release()
cv2.destroyAllWindows()