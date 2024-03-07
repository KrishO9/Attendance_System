# Face Recognition Real-Time Attendance System

## Introduction
Welcome to the Face Recognition Real-Time Attendance System! This project is designed to provide a seamless solution for real-time face recognition, specifically tailored for attendance tracking. By integrating face recognition technology with a real-time database, the system offers an efficient and accurate way to manage attendance records for students or employees.

## Tools and Packages Required
To successfully build and run this project, ensure the following tools and packages are installed:

- **Python**: The programming language used for the implementation.
- **Firebase**: A real-time database for storing student information and attendance records.
- **OpenCV**: A computer vision library used for webcam access and image processing.
- **dlib**: A toolkit for machine learning and computer vision.
- **face_recognition**: A face recognition library that is used to recognize and manipulate faces.
- **cvzone**: A computer vision library that provides additional functionality for face detection and tracking.

## Packages Required
Install the required Python packages using the following commands:

```bash
pip install opencv-python
pip install dlib
pip install face_recognition
pip install cvzone
pip install firebase-admin
```

## Building the Face Recognition Attendance System
### Step 1: Webcam Initialization
Initialize the webcam to capture real-time images. Configure the webcam parameters such as width and height accordingly.

```python
cap = cv2.VideoCapture(1)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height
```

### Step 2: Graphics and Modes
Load background images and modes for displaying information. Graphics, including background and modes, are crucial for creating a visually appealing interface.

```python
img_background = cv2.imread("resources/background.png")

img_mode_list = []
for path in os.listdir("resources/modes"):
    img_mode_list.append(cv2.imread(f"resources/modes/{path}"))
```

### Step 3: Student Information and Encoding Generation
Create a dictionary containing student information, including ID, name, year, department, total attendance, and last attendance. Resize and convert student images, and generate face encodings.

```python
data = {
    1: {"ID": 123, "name": "John Doe", "year": 3, "department": "Computer Science", "total_attendance": 0, "last_attendance": None},
    # Add information for other students
}

image_list = []
for ID, info in data.items():
    new_image = cv2.resize(info["image"], (216, 216))
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2BGR)
    image_list.append((ID, new_image))

encode_list_known = []
for ID, image in image_list:
    encode_list_known.append(face_recognition.face_encodings(image)[0])
```

### Step 4: Database Initialization
Initialize the Firebase database using credentials and set up references for storing student information and face encodings.

```python
cred = credentials.Certificate("path/to/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {"databaseURL": "your-database-url"})
ref_students = db.reference("students")
ref_encodings = db.reference("encodings")
```

### Step 5: Real-time Face Recognition
Implement real-time face recognition using OpenCV, dlib, and the face_recognition library. Track faces using cvzone and update the attendance database.

```python
while True:
    # Capture frame from webcam
    success, img = cap.read()

    # Detect faces
    faces = cvzone.face.detectFace(img)
    
    # Get face encodings
    encode_curr_list = []
    for face in faces:
        bbox = face["bbox"]
        cv2.rectangle(img, bbox, (255, 0, 255), 2)
        id = cvzone.face.getFaceID(img, bbox)
        encode_curr = face_recognition.face_encodings(img, [bbox])
        if encode_curr:
            encode_curr_list.append(encode_curr[0])

    # Compare with known faces
    for encode_curr in encode_curr_list:
        matches = face_recognition.compare_faces(encode_list_known, encode_curr)
        face_distance = face_recognition.face_distance(encode_list_known, encode_curr)
        match_index = np.argmin(face_distance)

        if matches[match_index]:
            student_id = list(data.keys())[match_index]
            # Update attendance logic here

    # Display the result
    cv2.imshow("Face Attendance", img)
    cv2.waitKey(1)
```
This is a simplified explanation, and the actual implementation might require additional considerations and adjustments based on specific project requirements.
