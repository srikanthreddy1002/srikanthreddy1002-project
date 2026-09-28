import cv2
from deepface import DeepFace
import os

if not os.path.exists("known"):
    os.makedirs("known")

cap = cv2.VideoCapture(0)
mode = "idle"

print("Press 's' to save face, 'r' to recognize, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF

    # 🔹 Save face
    if key == ord('s'):
        cv2.imwrite("known/face.jpg", frame)
        print("Face saved!")

    # 🔹 Start recognition
    elif key == ord('r'):
        mode = "recognize"
        print("Recognition mode ON")

    elif key == ord('q'):
        break

    # 🔹 Recognition mode
    if mode == "recognize":
        try:
            result = DeepFace.verify(
                img1_path="known/face.jpg",
                img2_path=frame,
                enforce_detection=False
            )

            if result["verified"]:
                text = "MATCH"
                color = (0, 255, 0)
            else:
                text = "UNKNOWN"
                color = (0, 0, 255)

            cv2.putText(frame, text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        except Exception as e:
            print("Error:", e)

    # ✅ SHOW FRAME AFTER DRAWING
    cv2.imshow("Face System", frame)

cap.release()
cv2.destroyAllWindows()
