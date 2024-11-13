# import cv2

# face_ref =cv2.CascadeClassifier("face_ref.xml")
# camera = cv2.VideoCapture(0)

# # Face detection function
# def face_detection(frame):
#     optimized_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minSize=(300, 300), minNeighbors=5)
#     return faces

# # Draw bounding box on the detected faces
# def drawer_box(frame):
#     for x, y, w, h in face_detection(frame):
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 4)

# # Function to close the window
# def close_window():
#     camera.release()
#     cv2.destroyAllWindows()
#     exit()

# # Main loop to display the video feed and detect faces
# def main():
#     while True:
#         _, frame = camera.read()
#         drawer_box(frame)
#         cv2.imshow("IzafFace AI", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             close_window()

# if __name__ == "__main__":
#     main()




# import cv2
# import time
# import threading

# # Load the face reference model
# face_ref = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# camera = cv2.VideoCapture(0)

# # Function to detect faces in the frame
# def face_detection(frame):
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_ref.detectMultiScale(gray_frame, scaleFactor=1.1, minSize=(100, 100), minNeighbors=5)
#     return faces

# # Function to draw bounding boxes around detected faces and show face count
# def draw_box(frame, faces):
#     for x, y, w, h in faces:
#         # Smooth rounded rectangle with border thickness and transparency effect
#         alpha = 0.4
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), -1)  # filled rectangle for transparency effect
#         cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # apply transparency
        
#         # Draw an outline for clarity
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

#     # Display the number of detected faces with shadow effect
#     cv2.putText(frame, f'Faces: {len(faces)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)  # shadow
#     cv2.putText(frame, f'Faces: {len(faces)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # text

# # Function to close the window and release camera properly
# def close_window():
#     camera.release()
#     cv2.destroyAllWindows()

# # Function to calculate frames per second (FPS)
# def calculate_fps(start_time, frame_count):
#     end_time = time.time()
#     fps = frame_count / (end_time - start_time)
#     return fps

# # Main loop to display video feed and detect faces
# def main():
#     start_time = time.time()
#     frame_count = 0

#     while True:
#         ret, frame = camera.read()
#         if not ret:
#             break

#         # Increment frame count
#         frame_count += 1

#         # Run face detection in a separate thread to avoid blocking
#         faces = []
#         thread = threading.Thread(target=lambda: faces.extend(face_detection(frame)))
#         thread.start()
#         thread.join()

#         # Draw boxes and face count
#         draw_box(frame, faces)

#         # Calculate and display FPS with a nice effect
#         fps = calculate_fps(start_time, frame_count)
#         cv2.putText(frame, f'FPS: {int(fps)}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)  # shadow
#         cv2.putText(frame, f'FPS: {int(fps)}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  # text

#         # Display user instructions
#         cv2.putText(frame, 'Press "Q" to quit', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
#         cv2.putText(frame, 'Press "Q" to quit', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

#         # Display the video feed
#         cv2.imshow("IzafFace AI - Face Detection", frame)

#         # Check for 'q' key to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             close_window()
#             break

# if __name__ == "__main__":
#     main()


# import cv2
# import time
# import threading
# import numpy as np

# # Load the face reference model
# face_ref = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# camera = cv2.VideoCapture(0)

# # Function to detect faces in the frame
# def face_detection(frame):
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_ref.detectMultiScale(gray_frame, scaleFactor=1.1, minSize=(100, 100), minNeighbors=5)
#     return faces

# # Function to draw a stylish bounding box with a dynamic color shift
# def draw_dynamic_box(frame, faces, frame_count):
#     for x, y, w, h in faces:
#         # Dynamic color shifting between red, green, and blue based on frame count
#         red = (frame_count % 255)
#         green = ((frame_count + 85) % 255)
#         blue = ((frame_count + 170) % 255)

#         # Create circular bounding box
#         center = (x + w // 2, y + h // 2)
#         radius = max(w // 2, h // 2)
#         cv2.circle(frame, center, radius, (red, green, blue), 4)

#         # Add some creative dots at the corners of the face
#         cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
#         cv2.circle(frame, (x + w, y), 8, (255, 0, 255), -1)
#         cv2.circle(frame, (x, y + h), 8, (255, 255, 0), -1)
#         cv2.circle(frame, (x + w, y + h), 8, (0, 255, 0), -1)

#     # Show the number of detected faces with a modern font style
#     cv2.putText(frame, f'Faces: {len(faces)}', (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

# # Function to close the window and release the camera properly
# def close_window():
#     camera.release()
#     cv2.destroyAllWindows()

# # Function to create a background gradient effect
# def apply_gradient(frame):
#     rows, cols, _ = frame.shape
#     gradient = np.linspace(0, 255, cols, dtype=np.uint8)
#     gradient = np.tile(gradient, (rows, 1))
#     gradient_bgr = cv2.merge([gradient, gradient[::-1], np.full_like(gradient, 100)])
#     return cv2.addWeighted(frame, 0.8, gradient_bgr, 0.5, 0)

# # Function to calculate frames per second (FPS)
# def calculate_fps(start_time, frame_count):
#     end_time = time.time()
#     fps = frame_count / (end_time - start_time)
#     return fps

# # Main loop to display video feed and detect faces
# def main():
#     start_time = time.time()
#     frame_count = 0

#     while True:
#         ret, frame = camera.read()
#         if not ret:
#             break

#         # Increment frame count
#         frame_count += 1

#         # Run face detection in a separate thread to avoid blocking
#         faces = []
#         thread = threading.Thread(target=lambda: faces.extend(face_detection(frame)))
#         thread.start()
#         thread.join()

#         # Apply a gradient effect for a visually engaging background
#         frame = apply_gradient(frame)

#         # Draw dynamic boxes and face count
#         draw_dynamic_box(frame, faces, frame_count)

#         # Calculate and display FPS with animation
#         fps = calculate_fps(start_time, frame_count)
#         fps_display = f'FPS: {int(fps)}'
#         text_size = cv2.getTextSize(fps_display, cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0]
#         position = (frame.shape[1] - text_size[0] - 10, 40)
#         cv2.putText(frame, fps_display, position, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

#         # Display user instructions
#         cv2.putText(frame, 'Press "Q" to quit', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

#         # Display the video feed
#         cv2.imshow("Creative Face Detection", frame)

#         # Check for 'q' key to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             close_window()
#             break

# if __name__ == "__main__":
#     main()



# Face Tracking Bubbles with Real-Time Emoji Reactions

# import cv2
# import time
# import numpy as np

# # Load the face reference model
# face_ref = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# camera = cv2.VideoCapture(0)

# # Function to detect faces in the frame
# def face_detection(frame):
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_ref.detectMultiScale(gray_frame, scaleFactor=1.1, minSize=(100, 100), minNeighbors=5)
#     return faces

# # Smoothly move "bubbles" toward detected face positions
# def move_bubble(current_pos, target_pos, smooth_factor=0.2):
#     x_current, y_current, w_current, h_current = current_pos
#     x_target, y_target, w_target, h_target = target_pos
#     new_x = int(x_current + smooth_factor * (x_target - x_current))
#     new_y = int(y_current + smooth_factor * (y_target - y_current))
#     new_w = int(w_current + smooth_factor * (w_target - w_current))
#     new_h = int(h_current + smooth_factor * (h_target - h_current))
#     return (new_x, new_y, new_w, new_h)

# # Function to draw face-tracking bubbles and emojis
# def draw_bubble_with_emoji(frame, faces, prev_faces, emojis, frame_count):
#     for i, (x, y, w, h) in enumerate(faces):
#         # Get previous face position or start from current
#         prev_face = prev_faces[i] if i < len(prev_faces) else (x, y, w, h)
#         # Smoothly move the bubble toward the new face position
#         new_pos = move_bubble(prev_face, (x, y, w, h))
#         x_new, y_new, w_new, h_new = new_pos

#         # Draw a playful bubble (transparent circle)
#         overlay = frame.copy()
#         cv2.circle(overlay, (x_new + w_new // 2, y_new + h_new // 2), max(w_new, h_new) // 2, (255, 255, 0), -1)
#         alpha = 0.3  # Transparency for bubble
#         cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

#         # Choose emoji based on face count
#         emoji = emojis[i % len(emojis)]
#         cv2.putText(frame, emoji, (x_new, y_new - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

# # Function to display an inverted mirror effect
# def apply_inverted_mirror(frame):
#     return cv2.flip(frame, 1)

# # Function to close the window and release camera properly
# def close_window():
#     camera.release()
#     cv2.destroyAllWindows()

# # Main loop to display video feed and detect faces
# def main():
#     prev_faces = []
#     frame_count = 0
#     emojis = ['😀', '😎', '😲', '😁', '😂']

#     while True:
#         ret, frame = camera.read()
#         if not ret:
#             break

#         # Increment frame count
#         frame_count += 1

#         # Apply the inverted mirror effect
#         frame = apply_inverted_mirror(frame)

#         # Run face detection
#         faces = face_detection(frame)

#         # Draw face-tracking bubbles and emojis
#         draw_bubble_with_emoji(frame, faces, prev_faces, emojis, frame_count)

#         # Update previous face positions
#         prev_faces = faces

#         # Display the video feed with effects
#         cv2.imshow("Face Tracking Bubbles & Emojis", frame)

#         # Check for 'q' key to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             close_window()
#             break

# if __name__ == "__main__":
#     main()



# Pro-Level Face Detection with "Motion Highlight" and Advanced FPS Counter

# import cv2
# import time

# # Load the face reference model
# face_ref = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# camera = cv2.VideoCapture(0)

# # Function to detect faces in the frame
# def face_detection(frame):
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_ref.detectMultiScale(gray_frame, scaleFactor=1.1, minSize=(100, 100), minNeighbors=5)
#     return faces

# # Function to calculate frames per second (FPS)
# def calculate_fps(start_time, frame_count):
#     end_time = time.time()
#     fps = frame_count / (end_time - start_time)
#     return fps

# # Function to highlight motion in the frame by detecting differences
# def motion_highlight(frame, prev_frame):
#     if prev_frame is None:
#         return frame
#     diff = cv2.absdiff(frame, prev_frame)
#     gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
#     mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
#     highlighted = cv2.addWeighted(frame, 1, mask, 0.5, 0)
#     return highlighted

# # Function to display additional UI elements (FPS and face count)
# def display_ui(frame, fps, face_count):
#     # Display FPS
#     cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    
#     # Display face count
#     cv2.putText(frame, f'Faces Detected: {face_count}', (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

# # Function to close the window and release the camera properly
# def close_window():
#     camera.release()
#     cv2.destroyAllWindows()
#     exit()

# # Main loop to display video feed and detect faces
# def main():
#     start_time = time.time()
#     frame_count = 0
#     prev_frame = None

#     while True:
#         ret, frame = camera.read()
#         if not ret:
#             break

#         # Increment frame count
#         frame_count += 1

#         # Detect faces
#         faces = face_detection(frame)

#         # Draw bounding boxes around detected faces
#         for x, y, w, h in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

#         # Highlight motion
#         frame = motion_highlight(frame, prev_frame)
#         prev_frame = frame.copy()

#         # Calculate FPS
#         fps = calculate_fps(start_time, frame_count)

#         # Display the user interface elements
#         display_ui(frame, fps, len(faces))

#         # Display the video feed
#         cv2.imshow("Pro Face Detection with Motion Highlight", frame)

#         # Check for 'q' key to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             close_window()

# if __name__ == "__main__":
#     main()




# keren

# import cv2
# import time

# # Load the face reference model
# face_ref = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# camera = cv2.VideoCapture(0)

# # Function to detect faces in the frame
# def face_detection(frame):
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_ref.detectMultiScale(gray_frame, scaleFactor=1.1, minSize=(100, 100), minNeighbors=5)
#     return faces

# # Function to calculate frames per second (FPS)
# def calculate_fps(start_time, frame_count):
#     end_time = time.time()
#     fps = frame_count / (end_time - start_time)
#     return fps

# # Function to draw bounding boxes and labels on faces
# def draw_box(frame, faces, track_ids):
#     for i, (x, y, w, h) in enumerate(faces):
#         # Draw a rounded rectangle
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
#         # Label with face tracking ID
#         cv2.putText(frame, f'ID {track_ids[i]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

# # Function to apply face blurring (privacy)
# def blur_faces(frame, faces):
#     for x, y, w, h in faces:
#         # Apply Gaussian blur to the detected face
#         face_region = frame[y:y + h, x:x + w]
#         blurred = cv2.GaussianBlur(face_region, (51, 51), 30)
#         frame[y:y + h, x:x + w] = blurred
#     return frame

# # Function to zoom into the first detected face (for a dynamic effect)
# def zoom_on_face(frame, faces):
#     if len(faces) > 0:
#         x, y, w, h = faces[0]  # Focus on the first face detected
#         face_roi = frame[y:y + h, x:x + w]
#         zoomed_face = cv2.resize(face_roi, None, fx=2, fy=2)  # Zoom in by 2x
#         return zoomed_face
#     return frame

# # Display additional UI elements (FPS, camera resolution, etc.)
# def display_ui(frame, fps, face_count):
#     # Display FPS
#     cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    
#     # Display face count
#     cv2.putText(frame, f'Faces Detected: {face_count}', (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

#     # Display camera resolution
#     resolution = f"Resolution: {int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
#     cv2.putText(frame, resolution, (10, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)

# # Function to close the window and release the camera properly
# def close_window():
#     camera.release()
#     cv2.destroyAllWindows()
#     exit()

# # Main loop to display video feed and detect faces
# def main():
#     start_time = time.time()
#     frame_count = 0
#     face_id = 0  # To track unique face IDs
#     track_ids = []  # List of tracking IDs for faces

#     while True:
#         ret, frame = camera.read()
#         if not ret:
#             break

#         # Increment frame count
#         frame_count += 1

#         # Detect faces
#         faces = face_detection(frame)

#         # Assign tracking IDs to faces (for simplicity, we assign sequential IDs)
#         while len(track_ids) < len(faces):
#             track_ids.append(face_id)
#             face_id += 1

#         # Draw bounding boxes around faces with tracking IDs
#         draw_box(frame, faces, track_ids)

#         # Optionally apply face blur (privacy)
#         blur_option = False
#         if blur_option:
#             frame = blur_faces(frame, faces)

#         # Optionally zoom in on the first detected face
#         zoom_option = False
#         if zoom_option:
#             zoomed_frame = zoom_on_face(frame, faces)
#             if zoomed_frame is not frame:
#                 frame = zoomed_frame

#         # Calculate FPS
#         fps = calculate_fps(start_time, frame_count)

#         # Display the user interface elements
#         display_ui(frame, fps, len(faces))

#         # Display the video feed
#         cv2.imshow("Izaf Face Detect Track and UI", frame)

#         # Check for 'q' key to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             close_window()

# if __name__ == "__main__":
#     main()





# import cv2
# import time
# import threading


# face_ref = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# camera = cv2.VideoCapture(0)

# lyrics_file = "song_lyrics.txt"


# def load_lyrics(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         lines = file.readlines()
#     return lines

# def face_detection(frame):
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_ref.detectMultiScale(gray_frame, scaleFactor=1.1, minSize=(100, 100), minNeighbors=5)
#     return faces

# def draw_box(frame, faces):
#     for x, y, w, h in faces:
#         # Rounded rectangle
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
#     # Display the number of detected faces
#     cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# def close_window():
#     camera.release()
#     cv2.destroyAllWindows()
#     exit()

# def calculate_fps(start_time, frame_count):
#     end_time = time.time()
#     fps = frame_count / (end_time - start_time)
#     return fps

# def display_lyrics(frame, lyrics_lines, start_time):
#     current_time = time.time() - start_time
#     line_index = int(current_time / 5) 

#     if line_index < len(lyrics_lines):
#         lyric_line = lyrics_lines[line_index].strip()
#     else:
#         lyric_line = "End of Song"

#     cv2.putText(frame, lyric_line, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# def main():
#     start_time = time.time()
#     frame_count = 0
#     lyrics_lines = load_lyrics(lyrics_file)
#     display_lyrics_flag = True 

#     while True:
#         ret, frame = camera.read()
#         if not ret:
#             break

#         frame_count += 1

#         faces = []
#         thread = threading.Thread(target=lambda: faces.extend(face_detection(frame)))
#         thread.start()
#         thread.join()

#         draw_box(frame, faces)

#         if display_lyrics_flag:
#             display_lyrics(frame, lyrics_lines, start_time)

#         # Calculate and display FPS
#         fps = calculate_fps(start_time, frame_count)
#         cv2.putText(frame, f'FPS: {int(fps)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         cv2.imshow("Face Detection with Synchronized Lyrics", frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('l'):  
#             display_lyrics_flag = not display_lyrics_flag

#     close_window()

# if __name__ == "__main__":
#     main()



#claudev1

import cv2
import time
import numpy as np
from datetime import datetime

# Load the face detection models - using both frontal and profile face detection for better coverage
face_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

# Initialize camera with higher resolution
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
camera.set(cv2.CAP_PROP_FPS, 60)  # Attempt to set higher FPS if camera supports it
camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available

# Enhanced face detection with both frontal and profile faces
def enhanced_face_detection(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Enhance image quality
    gray_frame = cv2.equalizeHist(gray_frame)
    
    # Detect both frontal and profile faces
    faces_frontal = face_frontal.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    faces_profile = face_profile.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Combine detected faces
    faces = np.vstack((faces_frontal, faces_profile)) if len(faces_frontal) and len(faces_profile) else \
           faces_frontal if len(faces_frontal) else faces_profile
    
    return faces

def calculate_fps(start_time, frame_count):
    end_time = time.time()
    fps = frame_count / (end_time - start_time)
    return round(fps, 1)  # Round to 1 decimal place

def draw_advanced_box(frame, faces, track_ids):
    for i, (x, y, w, h) in enumerate(faces):
        # Create a more aesthetic rounded rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add a semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y + h - 30), (x + w, y + h), (0, 255, 0), cv2.FILLED)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Add face ID and confidence score
        cv2.putText(frame, f'Face #{track_ids[i]}', (x + 5, y + h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def blur_faces_advanced(frame, faces):
    for x, y, w, h in faces:
        # Apply advanced privacy blur with adaptable kernel size
        kernel_size = max(w // 10, 1) * 2 + 1  # Ensure odd number
        face_region = frame[y:y + h, x:x + w]
        blurred = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 30)
        frame[y:y + h, x:x + w] = blurred
    return frame

def display_enhanced_ui(frame, fps, face_count, start_time):
    # Create semi-transparent overlay for UI
    ui_overlay = np.zeros_like(frame)
    
    # Add UI elements
    # FPS counter
    cv2.putText(ui_overlay, f'FPS: {fps}', (20, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    
    # Face count
    cv2.putText(ui_overlay, f'Faces: {face_count}', (20, 80),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    
    # Resolution
    resolution = f"Resolution: {int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    cv2.putText(ui_overlay, resolution, (20, 120),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    
    # Runtime
    elapsed_time = time.time() - start_time
    runtime = f"Runtime: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
    cv2.putText(ui_overlay, runtime, (20, 160),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    
    # Current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(ui_overlay, current_time, (frame.shape[1] - 250, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    
    # Blend UI overlay with main frame
    return cv2.addWeighted(frame, 1, ui_overlay, 0.7, 0)

def main():
    start_time = time.time()
    frame_count = 0
    face_id = 0
    track_ids = []
    blur_enabled = False
    
    print("Enhanced Face Detection System Started")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 'b' to toggle face blur")
    print("- Press 'f' to toggle fullscreen")
    
    fullscreen = False
    window_name = "Enhanced Face Detection System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Cannot read from camera")
            break

        frame_count += 1
        
        # Detect faces using enhanced detection
        faces = enhanced_face_detection(frame)
        
        # Update tracking IDs
        while len(track_ids) < len(faces):
            track_ids.append(face_id)
            face_id += 1
        
        # Apply effects
        if blur_enabled:
            frame = blur_faces_advanced(frame, faces)
        else:
            draw_advanced_box(frame, faces, track_ids)
        
        # Calculate and display FPS
        fps = calculate_fps(start_time, frame_count)
        
        # Add enhanced UI
        frame = display_enhanced_ui(frame, fps, len(faces), start_time)
        
        # Display the frame
        cv2.imshow(window_name, frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            blur_enabled = not blur_enabled
            print(f"Face blur: {'enabled' if blur_enabled else 'disabled'}")
        elif key == ord('f'):
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



#claude v2 - crash

# import cv2
# import numpy as np
# import torch
# import time
# from datetime import datetime
# import threading
# import queue
# import pygame
# from pygame import mixer

# class AdvancedDetectionSystem:
#     def __init__(self):
#         # Initialize YOLO model
#         self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#         self.model.conf = 0.35  # Confidence threshold
#         self.model.classes = None  # Detect all classes
        
#         # Initialize camera with high resolution
#         self.camera = cv2.VideoCapture(0)
#         self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#         self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#         self.camera.set(cv2.CAP_PROP_FPS, 60)
        
#         # Initialize UI elements
#         self.window_name = "Advanced Detection System"
#         cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
#         # Detection settings
#         self.detection_modes = {
#             'all': True,           # All objects
#             'person': True,        # People only
#             'vehicle': True,       # Vehicles only
#             'animal': True         # Animals only
#         }
        
#         # UI settings
#         self.show_fps = True
#         self.show_counts = True
#         self.dark_mode = True
#         self.alert_mode = False
#         self.recording = False
        
#         # Initialize sound effects
#         # pygame.mixer.init()
#         # self.detection_sound = pygame.mixer.Sound('detection_beep.wav')  # You'll need to provide this sound file
        
#         # Initialize video writer
#         self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         self.out = None
        
#         # Stats and counters
#         self.start_time = time.time()
#         self.frame_count = 0
#         self.fps = 0
#         self.detection_history = []
        
#         # Threading setup
#         self.frame_queue = queue.Queue(maxsize=10)
#         self.result_queue = queue.Queue(maxsize=10)
#         self.running = True
        
#         # Color schemes
#         self.color_schemes = {
#             'light': {
#                 'background': (255, 255, 255),
#                 'text': (0, 0, 0),
#                 'accent': (0, 120, 255)
#             },
#             'dark': {
#                 'background': (0, 0, 0),
#                 'text': (255, 255, 255),
#                 'accent': (0, 255, 200)
#             }
#         }
        
#         # Initialize detection thread
#         self.detection_thread = threading.Thread(target=self._detection_worker)
#         self.detection_thread.start()

#     def _detection_worker(self):
#         """Worker thread for object detection"""
#         while self.running:
#             if not self.frame_queue.empty():
#                 frame = self.frame_queue.get()
#                 results = self.model(frame)
#                 self.result_queue.put(results)

#     def draw_ui_overlay(self, frame, detections):
#         """Draw advanced UI overlay with detection information"""
#         # Create semi-transparent overlay
#         overlay = np.zeros_like(frame)
#         colors = self.color_schemes['dark'] if self.dark_mode else self.color_schemes['light']
        
#         # Top bar
#         cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), colors['background'], -1)
        
#         # Bottom bar
#         cv2.rectangle(overlay, (0, frame.shape[0]-60), (frame.shape[1], frame.shape[0]), 
#                      colors['background'], -1)
        
#         # Display FPS
#         if self.show_fps:
#             fps_text = f"FPS: {int(self.fps)}"
#             cv2.putText(frame, fps_text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 
#                        1, colors['text'], 2)
        
#         # Display detection counts
#         if self.show_counts:
#             counts = self._get_detection_counts(detections)
#             count_text = " | ".join([f"{k}: {v}" for k, v in counts.items()])
#             cv2.putText(frame, count_text, (200, 40), cv2.FONT_HERSHEY_DUPLEX, 
#                        0.7, colors['text'], 2)
        
#         # Display current time
#         time_text = datetime.now().strftime("%H:%M:%S")
#         cv2.putText(frame, time_text, (frame.shape[1]-150, 40), 
#                    cv2.FONT_HERSHEY_DUPLEX, 0.7, colors['text'], 2)
        
#         # Display recording indicator
#         if self.recording:
#             cv2.circle(frame, (frame.shape[1]-180, 30), 10, (0, 0, 255), -1)
        
#         # Blend overlay with frame
#         return cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

#     def draw_detection_boxes(self, frame, detections):
#         """Draw aesthetic detection boxes with labels"""
#         for det in detections.xyxy[0]:
#             x1, y1, x2, y2, conf, cls = det.cpu().numpy()
#             label = f"{detections.names[int(cls)]} {conf:.2f}"
            
#             # Create gradient effect for box
#             color = self._get_color_for_class(detections.names[int(cls)])
            
#             # Draw box with rounded corners
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
#             # Draw label background
#             label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
#             cv2.rectangle(frame, (int(x1), int(y1)-25), 
#                          (int(x1)+label_size[0], int(y1)), color, -1)
            
#             # Draw label text
#             cv2.putText(frame, label, (int(x1), int(y1)-5), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#     def _get_color_for_class(self, class_name):
#         """Get consistent color for object class"""
#         color_map = {
#             'person': (0, 255, 0),
#             'car': (255, 0, 0),
#             'truck': (255, 128, 0),
#             'dog': (255, 255, 0),
#             'cat': (0, 255, 255)
#         }
#         return color_map.get(class_name, (128, 128, 128))

#     def _get_detection_counts(self, detections):
#         """Count detections by class"""
#         counts = {}
#         for det in detections.xyxy[0]:
#             cls_name = detections.names[int(det[5])]
#             counts[cls_name] = counts.get(cls_name, 0) + 1
#         return counts

#     def toggle_recording(self):
#         """Toggle video recording"""
#         self.recording = not self.recording
#         if self.recording:
#             filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
#             self.out = cv2.VideoWriter(filename, self.fourcc, 20.0, 
#                                      (int(self.camera.get(3)), int(self.camera.get(4))))
#         else:
#             if self.out:
#                 self.out.release()
#                 self.out = None

#     def process_frame(self, frame):
#         """Process a single frame"""
#         # Queue frame for detection
#         if self.frame_queue.qsize() < self.frame_queue.maxsize:
#             self.frame_queue.put(frame)
        
#         # Get detection results if available
#         if not self.result_queue.empty():
#             detections = self.result_queue.get()
            
#             # Apply UI overlay
#             frame = self.draw_ui_overlay(frame, detections)
            
#             # Draw detection boxes
#             self.draw_detection_boxes(frame, detections)
            
#             # Record if enabled
#             if self.recording and self.out:
#                 self.out.write(frame)
            
#             # Play alert sound if needed
#             if self.alert_mode and len(detections.xyxy[0]) > 0:
#                 self.detection_sound.play()
        
#         return frame

#     def run(self):
#         """Main loop"""
#         print("Advanced Detection System Started")
#         print("Controls:")
#         print("- 'q': Quit")
#         print("- 'r': Toggle recording")
#         print("- 'm': Toggle dark/light mode")
#         print("- 'a': Toggle alert mode")
#         print("- 'f': Toggle fullscreen")
#         print("- 's': Save screenshot")

#         try:
#             while self.running:
#                 ret, frame = self.camera.read()
#                 if not ret:
#                     break

#                 self.frame_count += 1
#                 self.fps = self.frame_count / (time.time() - self.start_time)

#                 # Process frame
#                 processed_frame = self.process_frame(frame)
                
#                 # Display frame
#                 cv2.imshow(self.window_name, processed_frame)
                
#                 # Handle keyboard input
#                 key = cv2.waitKey(1) & 0xFF
#                 if key == ord('q'):
#                     self.running = False
#                 elif key == ord('r'):
#                     self.toggle_recording()
#                 elif key == ord('m'):
#                     self.dark_mode = not self.dark_mode
#                 elif key == ord('a'):
#                     self.alert_mode = not self.alert_mode
#                 elif key == ord('f'):
#                     cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, 
#                                         cv2.WINDOW_FULLSCREEN)
#                 elif key == ord('s'):
#                     filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#                     cv2.imwrite(filename, processed_frame)

#         finally:
#             self.cleanup()

#     def cleanup(self):
#         """Cleanup resources"""
#         self.running = False
#         self.detection_thread.join()
#         if self.out:
#             self.out.release()
#         self.camera.release()
#         cv2.destroyAllWindows()
#         pygame.mixer.quit()

# if __name__ == "__main__":
#     detector = AdvancedDetectionSystem()
#     detector.run()




# claude v2 norm but lag

# import cv2
# import numpy as np
# import torch
# import time
# from datetime import datetime
# import threading
# import queue
# import pygame
# from pygame import mixer
# import os
# import logging
# from pathlib import Path

# class AdvancedDetectionSystem:
#     def __init__(self):
#         # Set up logging
#         self._setup_logging()
        
#         try:
#             # Initialize YOLO model with error handling
#             self.logger.info("Initializing YOLO model...")
#             self.model = self._initialize_model()
            
#             # Initialize camera
#             self.logger.info("Initializing camera...")
#             self.camera = self._initialize_camera()
            
#             if not self.camera.isOpened():
#                 raise RuntimeError("Failed to open camera")
            
#             # Create output directory
#             self.output_dir = Path("output")
#             self.output_dir.mkdir(exist_ok=True)
            
#             # Initialize UI elements
#             self.window_name = "Advanced Detection System"
#             cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            
#             # Detection settings
#             self.detection_modes = {
#                 'all': True,
#                 'person': True,
#                 'vehicle': True,
#                 'animal': True
#             }
            
#             # UI settings
#             self.show_fps = True
#             self.show_counts = True
#             self.dark_mode = True
#             self.alert_mode = False
#             self.recording = False
#             self.fullscreen = False
            
#             # Initialize sound if available
#             self._initialize_sound()
            
#             # Initialize video writer settings
#             self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
#             self.out = None
            
#             # Stats and counters
#             self.start_time = time.time()
#             self.frame_count = 0
#             self.fps = 0
#             self.detection_history = []
            
#             # Threading setup with proper synchronization
#             self.frame_queue = queue.Queue(maxsize=10)
#             self.result_queue = queue.Queue(maxsize=10)
#             self.running = True
#             self.thread_lock = threading.Lock()
            
#             # Color schemes
#             self.color_schemes = {
#                 'light': {
#                     'background': (255, 255, 255),
#                     'text': (0, 0, 0),
#                     'accent': (0, 120, 255)
#                 },
#                 'dark': {
#                     'background': (0, 0, 0),
#                     'text': (255, 255, 255),
#                     'accent': (0, 255, 200)
#                 }
#             }
            
#             # Start detection thread
#             self.detection_thread = threading.Thread(target=self._detection_worker)
#             self.detection_thread.daemon = True
#             self.detection_thread.start()
            
#         except Exception as e:
#             self.logger.error(f"Initialization error: {str(e)}")
#             raise

#     def _setup_logging(self):
#         """Set up logging configuration"""
#         self.logger = logging.getLogger('AdvancedDetectionSystem')
#         self.logger.setLevel(logging.INFO)
        
#         formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
#         # Console handler
#         ch = logging.StreamHandler()
#         ch.setFormatter(formatter)
#         self.logger.addHandler(ch)
        
#         # File handler
#         fh = logging.FileHandler('detection_system.log')
#         fh.setFormatter(formatter)
#         self.logger.addHandler(fh)

#     def _initialize_model(self):
#         """Initialize YOLO model with error handling"""
#         try:
#             model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#             model.conf = 0.35
#             model.classes = None
#             return model
#         except Exception as e:
#             self.logger.error(f"Failed to initialize YOLO model: {str(e)}")
#             raise

#     def _initialize_camera(self):
#         """Initialize camera with error handling"""
#         camera = cv2.VideoCapture(0)
#         if not camera.isOpened():
#             self.logger.error("Failed to open camera")
#             raise RuntimeError("Cannot open camera")
            
#         # Set camera properties with error checking
#         camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#         camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#         camera.set(cv2.CAP_PROP_FPS, 60)
        
#         return camera

#     def _initialize_sound(self):
#         """Initialize sound system with error handling"""
#         try:
#             pygame.mixer.init()
#             sound_file = Path("detection_beep.wav")
#             if sound_file.exists():
#                 self.detection_sound = pygame.mixer.Sound(str(sound_file))
#             else:
#                 self.logger.warning("Sound file not found. Alert sounds disabled.")
#                 self.detection_sound = None
#         except Exception as e:
#             self.logger.warning(f"Failed to initialize sound: {str(e)}")
#             self.detection_sound = None

#     def _detection_worker(self):
#         """Worker thread for object detection"""
#         while self.running:
#             try:
#                 if not self.frame_queue.empty():
#                     frame = self.frame_queue.get(timeout=1)
#                     with torch.no_grad():
#                         results = self.model(frame)
#                     self.result_queue.put(results)
#             except queue.Empty:
#                 continue
#             except Exception as e:
#                 self.logger.error(f"Detection error: {str(e)}")
#                 continue

#     def draw_ui_overlay(self, frame, detections):
#         """Draw advanced UI overlay with detection information"""
#         try:
#             # Create semi-transparent overlay
#             overlay = np.zeros_like(frame)
#             colors = self.color_schemes['dark'] if self.dark_mode else self.color_schemes['light']
            
#             # Draw top and bottom bars
#             cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), colors['background'], -1)
#             cv2.rectangle(overlay, (0, frame.shape[0]-60), (frame.shape[1], frame.shape[0]), 
#                          colors['background'], -1)
            
#             # Display FPS
#             if self.show_fps:
#                 fps_text = f"FPS: {int(self.fps)}"
#                 cv2.putText(frame, fps_text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 
#                            1, colors['text'], 2)
            
#             # Display detection counts
#             if self.show_counts and detections is not None:
#                 counts = self._get_detection_counts(detections)
#                 count_text = " | ".join([f"{k}: {v}" for k, v in counts.items()])
#                 cv2.putText(frame, count_text, (200, 40), cv2.FONT_HERSHEY_DUPLEX, 
#                            0.7, colors['text'], 2)
            
#             # Display current time
#             time_text = datetime.now().strftime("%H:%M:%S")
#             cv2.putText(frame, time_text, (frame.shape[1]-150, 40), 
#                        cv2.FONT_HERSHEY_DUPLEX, 0.7, colors['text'], 2)
            
#             # Display recording indicator
#             if self.recording:
#                 cv2.circle(frame, (frame.shape[1]-180, 30), 10, (0, 0, 255), -1)
            
#             # Blend overlay with frame
#             return cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
            
#         except Exception as e:
#             self.logger.error(f"UI overlay error: {str(e)}")
#             return frame

#     def draw_detection_boxes(self, frame, detections):
#         """Draw aesthetic detection boxes with labels"""
#         try:
#             if detections is None or len(detections.xyxy) == 0:
#                 return
                
#             for det in detections.xyxy[0]:
#                 x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                
#                 # Ensure coordinates are within frame boundaries
#                 x1 = max(0, min(frame.shape[1], int(x1)))
#                 x2 = max(0, min(frame.shape[1], int(x2)))
#                 y1 = max(0, min(frame.shape[0], int(y1)))
#                 y2 = max(0, min(frame.shape[0], int(y2)))
                
#                 label = f"{detections.names[int(cls)]} {conf:.2f}"
#                 color = self._get_color_for_class(detections.names[int(cls)])
                
#                 # Draw box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
#                 # Draw label background and text
#                 label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
#                 cv2.rectangle(frame, (x1, y1-25), (x1+label_size[0], y1), color, -1)
#                 cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
#                            0.6, (255, 255, 255), 2)
                
#         except Exception as e:
#             self.logger.error(f"Detection box drawing error: {str(e)}")

#     def toggle_recording(self):
#         """Toggle video recording with error handling"""
#         try:
#             self.recording = not self.recording
#             if self.recording:
#                 filename = self.output_dir / f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
#                 self.out = cv2.VideoWriter(str(filename), self.fourcc, 20.0, 
#                                          (int(self.camera.get(3)), int(self.camera.get(4))))
#                 self.logger.info(f"Started recording to {filename}")
#             else:
#                 if self.out:
#                     self.out.release()
#                     self.out = None
#                     self.logger.info("Stopped recording")
                    
#         except Exception as e:
#             self.logger.error(f"Recording error: {str(e)}")
#             self.recording = False
#             if self.out:
#                 self.out.release()
#                 self.out = None

#     def process_frame(self, frame):
#         """Process a single frame with error handling"""
#         try:
#             # Queue frame for detection
#             if self.frame_queue.qsize() < self.frame_queue.maxsize:
#                 self.frame_queue.put(frame.copy())
            
#             # Get detection results if available
#             detections = None
#             if not self.result_queue.empty():
#                 detections = self.result_queue.get_nowait()
                
#                 # Apply UI overlay
#                 frame = self.draw_ui_overlay(frame, detections)
                
#                 # Draw detection boxes
#                 self.draw_detection_boxes(frame, detections)
                
#                 # Record if enabled
#                 if self.recording and self.out:
#                     self.out.write(frame)
                
#                 # Play alert sound if needed
#                 if self.alert_mode and self.detection_sound and len(detections.xyxy[0]) > 0:
#                     self.detection_sound.play()
            
#             return frame
            
#         except Exception as e:
#             self.logger.error(f"Frame processing error: {str(e)}")
#             return frame

#     def run(self):
#         """Main loop with error handling"""
#         self.logger.info("Advanced Detection System Started")
#         self._print_controls()
        
#         try:
#             while self.running:
#                 ret, frame = self.camera.read()
#                 if not ret:
#                     self.logger.error("Failed to read frame from camera")
#                     break

#                 self.frame_count += 1
#                 self.fps = self.frame_count / (time.time() - self.start_time)

#                 # Process frame
#                 processed_frame = self.process_frame(frame)
                
#                 # Display frame
#                 cv2.imshow(self.window_name, processed_frame)
                
#                 # Handle keyboard input
#                 self._handle_keyboard_input()
                
#         except KeyboardInterrupt:
#             self.logger.info("Received keyboard interrupt")
#         except Exception as e:
#             self.logger.error(f"Runtime error: {str(e)}")
#         finally:
#             self.cleanup()

#     def _handle_keyboard_input(self):
#         """Handle keyboard input with error handling"""
#         try:
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 self.running = False
#             elif key == ord('r'):
#                 self.toggle_recording()
#             elif key == ord('m'):
#                 self.dark_mode = not self.dark_mode
#             elif key == ord('a'):
#                 self.alert_mode = not self.alert_mode
#             elif key == ord('f'):
#                 self.fullscreen = not self.fullscreen
#                 if self.fullscreen:
#                     cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, 
#                                         cv2.WINDOW_FULLSCREEN)
#                 else:
#                     cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, 
#                                         cv2.WINDOW_NORMAL)
#             elif key == ord('s'):
#                 self._save_screenshot()
                
#         except Exception as e:
#             self.logger.error(f"Keyboard input error: {str(e)}")

#     def _save_screenshot(self):
#         """Save screenshot with error handling"""
#         try:
#             filename = self.output_dir / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#             ret, frame = self.camera.read()
#             if ret:
#                 cv2.imwrite(str(filename), frame)
#                 self.logger.info(f"Screenshot saved: {filename}")
#             else:
#                 self.logger.error("Failed to capture screenshot")
                
#         except Exception as e:
#             self.logger.error(f"Screenshot error: {str(e)}")

#     def _print_controls(self):
#         """Print control information"""
#         controls = [
#             "Controls:",
#             "- 'q': Quit",
#             "- 'r': Toggle recording",
#             "- 'm': Toggle dark/light mode",
#             "- 'a': Toggle alert mode",
#             "- 'f': Toggle fullscreen",
#             "- 's': Save screenshot"
#         ]
#         for control in controls:
#             self.logger.info(control)

#     def cleanup(self):
#         """Cleanup resources with error handling"""
#         self.logger.info("Cleaning up resources...")
#         self.running = False
        
#         try:
#             # Stop detection thread
#             if hasattr(self, 'detection_thread') and self.detection_thread.is_alive():
#                 self.detection_thread.join(timeout=1.0)
            
#             # Release video writer
#             if self.out:
#                 self.out.release()
            
#             # Release camera
#             if hasattr(self, 'camera'):
#                 self.camera.release()
            
#             # Cleanup OpenCV windows
#             cv2.destroyAllWindows()
            
#             # Cleanup pygame
#             if pygame.mixer.get_init():
#                 pygame.mixer.quit()
                
#             self.logger.info("Cleanup completed successfully")
            
#         except Exception as e:
#             self.logger.error(f"Cleanup error: {str(e)}")
#         finally:
#             # Close logging handlers
#             for handler in self.logger.handlers[:]:
#                 handler.close()
#                 self.logger.removeHandler(handler)

#     def _get_color_for_class(self, class_name):
#         """Get consistent color for object class"""
#         color_map = {
#             'person': (0, 255, 0),
#             'car': (255, 0, 0),
#             'truck': (255, 128, 0),
#             'motorcycle': (255, 64, 0),
#             'bicycle': (255, 192, 0),
#             'bus': (255, 96, 0),
#             'dog': (255, 255, 0),
#             'cat': (0, 255, 255),
#             'horse': (128, 255, 0),
#             'sheep': (0, 255, 128),
#             'cow': (0, 255, 192),
#             'elephant': (0, 192, 255),
#             'bear': (0, 128, 255),
#             'zebra': (0, 64, 255),
#             'giraffe': (128, 0, 255)
#         }
#         return color_map.get(class_name, (128, 128, 128))

#     def _get_detection_counts(self, detections):
#         """Count detections by class with error handling"""
#         counts = {}
#         try:
#             if detections is not None and hasattr(detections, 'xyxy') and len(detections.xyxy) > 0:
#                 for det in detections.xyxy[0]:
#                     cls_name = detections.names[int(det[5])]
#                     counts[cls_name] = counts.get(cls_name, 0) + 1
#         except Exception as e:
#             self.logger.error(f"Error counting detections: {str(e)}")
#         return counts

#     def toggle_detection_mode(self, mode):
#         """Toggle specific detection mode"""
#         try:
#             if mode in self.detection_modes:
#                 self.detection_modes[mode] = not self.detection_modes[mode]
#                 self.logger.info(f"Detection mode '{mode}' {'enabled' if self.detection_modes[mode] else 'disabled'}")
#             else:
#                 self.logger.warning(f"Invalid detection mode: {mode}")
#         except Exception as e:
#             self.logger.error(f"Error toggling detection mode: {str(e)}")

#     def set_model_confidence(self, confidence):
#         """Set model confidence threshold with validation"""
#         try:
#             if 0.0 <= confidence <= 1.0:
#                 self.model.conf = confidence
#                 self.logger.info(f"Model confidence threshold set to {confidence}")
#             else:
#                 self.logger.warning("Confidence threshold must be between 0.0 and 1.0")
#         except Exception as e:
#             self.logger.error(f"Error setting model confidence: {str(e)}")

#     def get_system_status(self):
#         """Get current system status"""
#         try:
#             status = {
#                 'fps': self.fps,
#                 'frame_count': self.frame_count,
#                 'running_time': time.time() - self.start_time,
#                 'recording': self.recording,
#                 'dark_mode': self.dark_mode,
#                 'alert_mode': self.alert_mode,
#                 'detection_modes': self.detection_modes.copy(),
#                 'model_confidence': self.model.conf,
#                 'frame_queue_size': self.frame_queue.qsize(),
#                 'result_queue_size': self.result_queue.qsize()
#             }
#             return status
#         except Exception as e:
#             self.logger.error(f"Error getting system status: {str(e)}")
#             return {}

#     @staticmethod
#     def get_supported_classes():
#         """Get list of supported object classes"""
#         try:
#             model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#             return list(model.names.values())
#         except Exception as e:
#             logging.error(f"Error getting supported classes: {str(e)}")
#             return []

#     def export_detection_history(self, filename):
#         """Export detection history to CSV file"""
#         try:
#             if not self.detection_history:
#                 self.logger.warning("No detection history to export")
#                 return False

#             import csv
#             filepath = self.output_dir / filename
#             with open(filepath, 'w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(['timestamp', 'class', 'confidence', 'box_coordinates'])
#                 for detection in self.detection_history:
#                     writer.writerow([
#                         detection['timestamp'],
#                         detection['class'],
#                         detection['confidence'],
#                         detection['box_coordinates']
#                     ])
#             self.logger.info(f"Detection history exported to {filepath}")
#             return True
#         except Exception as e:
#             self.logger.error(f"Error exporting detection history: {str(e)}")
#             return False

# def main():
#     """Main entry point with error handling"""
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
    
#     try:
#         logger.info("Starting Advanced Detection System...")
#         detector = AdvancedDetectionSystem()
#         detector.run()
#     except KeyboardInterrupt:
#         logger.info("Application terminated by user")
#     except Exception as e:
#         logger.error(f"Application error: {str(e)}")
#     finally:
#         logger.info("Application shutdown complete")

# if __name__ == "__main__":
#     main()



#problem

# import cv2
# import numpy as np
# import torch
# import mediapipe as mp
# import tensorflow as tf
# import pygame
# import json
# import os
# import threading
# import queue
# from datetime import datetime
# from pathlib import Path
# from PIL import Image
# import pandas as pd
# import matplotlib.pyplot as plt
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# import warnings
# warnings.filterwarnings('ignore')

# class UltimateDetectionSystem:
#     def __init__(self):
#         self.setup_directories()
#         self.initialize_models()
#         self.setup_interface()
#         # self.initialize_tracking()
#         self.setup_analysis()
        
#     def setup_directories(self):
#         """Setup directory structure for saving data"""
#         self.base_dir = Path("detection_system_data")
#         self.dirs = {
#             'recordings': self.base_dir / 'recordings',
#             'screenshots': self.base_dir / 'screenshots',
#             'analytics': self.base_dir / 'analytics',
#             'models': self.base_dir / 'models'
#         }
#         for dir_path in self.dirs.values():
#             dir_path.mkdir(parents=True, exist_ok=True)

#     def initialize_models(self):
#         """Initialize all detection models"""
#         # YOLO for general object detection
#         self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
#         self.yolo.conf = 0.4
        
#         # MediaPipe for detailed face and pose detection
#         self.mp_face = mp.solutions.face_mesh.FaceMesh(
#             max_num_faces=10,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
#         self.mp_pose = mp.solutions.pose.Pose(
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
        
#         # MobileNetV2 for additional classification
#         self.mobilenet = MobileNetV2(weights='imagenet', include_top=True)
        
#         # Load emotion detection model (you'll need to provide this)
#         # self.emotion_model = tf.keras.models.load_model('emotion_model.h5', compile=False)
        
#     def setup_interface(self):
#         """Initialize display and UI elements"""
#         # Initialize Pygame for advanced UI
#         pygame.init()
#         pygame.mixer.init()
        
#         # Camera setup
#         self.camera = cv2.VideoCapture(0)
#         self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#         self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#         self.camera.set(cv2.CAP_PROP_FPS, 60)
#         self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
#         self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        
#         # UI settings
#         self.window_name = "Ultimate Detection System"
#         cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
#         # Interface modes
#         self.modes = {
#             'general': True,      # General object detection
#             'face': True,         # Face analysis
#             'pose': True,         # Pose estimation
#             'emotion': True,      # Emotion detection
#             'tracking': True,     # Object tracking
#             'analysis': True      # Real-time analytics
#         }
        
#         # UI themes
#         self.themes = {
#             'dark': {
#                 'background': (15, 15, 15),
#                 'primary': (0, 120, 255),
#                 'secondary': (0, 255, 200),
#                 'text': (255, 255, 255),
#                 'accent': (255, 128, 0)
#             },
#             'light': {
#                 'background': (240, 240, 240),
#                 'primary': (0, 90, 255),
#                 'secondary': (0, 180, 150),
#                 'text': (0, 0, 0),
#                 'accent': (255, 100, 0)
#             }
#         }
#         self.current_theme = 'dark'
        
#         # Load sound effects
#         # self.sounds = {
#         #     'detection': pygame.mixer.Sound('sounds/detection.wav'),
#         #     'alert': pygame.mixer.Sound('sounds/alert.wav'),
#         #     'snapshot': pygame.mixer.Sound('sounds/snapshot.wav')
#         # }
        
#     def initialize_tracking(self):
#         """Initialize object tracking systems"""
#         self.tracker = cv2.TrackerCSRT_create()
#         self.tracking_history = []
#         self.detected_objects = {}
#         self.object_counts = {}
#         self.detection_queue = queue.Queue(maxsize=30)
#         self.result_queue = queue.Queue(maxsize=30)
        
#         # Start tracking threads
#         self.running = True
#         # self.tracking_thread = threading.Thread(target=self._tracking_worker)
#         self.analysis_thread = threading.Thread(target=self._analysis_worker)
#         self.tracking_thread.start()
#         self.analysis_thread.start()

#     def setup_analysis(self):
#         """Setup analysis and logging systems"""
#         self.analytics = {
#             'detections': [],
#             'fps_history': [],
#             'object_frequency': {},
#             'interaction_zones': {},
#             'motion_heatmap': np.zeros((1080, 1920), dtype=np.float32)
#         }
        
#         # Initialize CSV logger
#         self.csv_logger = pd.DataFrame(columns=[
#             'timestamp', 'object_type', 'confidence',
#             'position_x', 'position_y', 'action'
#         ])

#     def process_frame(self, frame):
#         """Main frame processing pipeline"""
#         processed_frame = frame.copy()
        
#         # Apply AI models based on active modes
#         if self.modes['general']:
#             processed_frame = self.apply_object_detection(processed_frame)
        
#         if self.modes['face']:
#             processed_frame = self.apply_face_analysis(processed_frame)
            
#         if self.modes['pose']:
#             processed_frame = self.apply_pose_estimation(processed_frame)
            
#         if self.modes['emotion']:
#             processed_frame = self.apply_emotion_detection(processed_frame)
            
#         if self.modes['tracking']:
#             processed_frame = self.apply_object_tracking(processed_frame)
            
#         # Apply UI overlay
#         processed_frame = self.apply_ui_overlay(processed_frame)
        
#         return processed_frame

#     def apply_object_detection(self, frame):
#         """Apply YOLO object detection"""
#         results = self.yolo(frame)
        
#         # Process detections
#         for det in results.xyxy[0]:
#             x1, y1, x2, y2, conf, cls = det.cpu().numpy()
#             label = f"{results.names[int(cls)]} {conf:.2f}"
            
#             # Draw aesthetic box
#             self.draw_detection_box(frame, (int(x1), int(y1), int(x2), int(y2)), 
#                                   label, conf)
            
#             # Update analytics
#             self.update_detection_analytics(results.names[int(cls)], conf, 
#                                          (x1, y1, x2, y2))
        
#         return frame

#     def apply_face_analysis(self, frame):
#         """Apply face mesh and analysis"""
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.mp_face.process(rgb_frame)
        
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 # Draw face mesh
#                 self.draw_face_mesh(frame, face_landmarks)
                
#                 # Analyze facial features
#                 self.analyze_facial_features(frame, face_landmarks)
        
#         return frame

#     def apply_pose_estimation(self, frame):
#         """Apply pose estimation"""
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.mp_pose.process(rgb_frame)
        
#         if results.pose_landmarks:
#             # Draw pose landmarks
#             self.draw_pose_landmarks(frame, results.pose_landmarks)
            
#             # Analyze pose
#             self.analyze_pose(results.pose_landmarks)
        
#         return frame

#     def apply_emotion_detection(self, frame):
#         """Apply emotion detection"""
#         faces = cv2.CascadeClassifier(cv2.data.haarcascades + 
#                                     'haarcascade_frontalface_default.xml')\
#                 .detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 4)
        
#         for (x, y, w, h) in faces:
#             face_img = frame[y:y+h, x:x+w]
#             face_img = cv2.resize(face_img, (48, 48))
#             face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
#             face_img = np.expand_dims(face_img, axis=[0, -1])
            
#             emotion = self.emotion_model.predict(face_img)
#             emotion_label = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprised'][
#                 np.argmax(emotion)]
            
#             self.draw_emotion_label(frame, (x, y), emotion_label)
        
#         return frame

#     def draw_detection_box(self, frame, bbox, label, confidence):
#         """Draw aesthetic detection box with label"""
#         x1, y1, x2, y2 = bbox
#         color = self.get_color_by_confidence(confidence)
        
#         # Draw gradient box
#         gradient = np.linspace(0, 1, 10)
#         for i in range(3):
#             cv2.rectangle(frame, (x1+i, y1+i), (x2-i, y2-i), color, 1)
            
#         # Draw label background
#         label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
#         cv2.rectangle(frame, (x1, y1-30), (x1+label_size[0], y1), color, -1)
        
#         # Draw label
#         cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.6, (255, 255, 255), 2)

#     def apply_ui_overlay(self, frame):
#         """Apply advanced UI overlay"""
#         # Create base overlay
#         overlay = np.zeros_like(frame)
#         theme = self.themes[self.current_theme]
        
#         # Top bar
#         cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), 
#                      theme['background'], -1)
        
#         # Bottom bar
#         cv2.rectangle(overlay, (0, frame.shape[0]-60), 
#                      (frame.shape[1], frame.shape[0]), theme['background'], -1)
        
#         # Add statistics
#         self.draw_statistics(overlay, theme)
        
#         # Add mode indicators
#         self.draw_mode_indicators(overlay, theme)
        
#         # Add system status
#         self.draw_system_status(overlay, theme)
        
#         # Blend overlay
#         return cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)

#     def run(self):
#         """Main loop"""
#         print("Ultimate Detection System Started")
#         self.print_controls()
        
#         try:
#             while self.running:
#                 ret, frame = self.camera.read()
#                 if not ret:
#                     break
                
#                 # Process frame
#                 processed_frame = self.process_frame(frame)
                
#                 # Display frame
#                 cv2.imshow(self.window_name, processed_frame)
                
#                 # Handle keyboard input
#                 self.handle_input()
                
#                 # Update analytics
#                 self.update_analytics()
                
#         finally:
#             self.cleanup()

#     def handle_input(self):
#         """Handle keyboard and mouse input"""
#         key = cv2.waitKey(1) & 0xFF
        
#         # Key mappings
#         key_actions = {
#             'q': self.quit,
#             'r': self.toggle_recording,
#             'm': self.cycle_modes,
#             't': self.cycle_themes,
#             'a': self.toggle_analytics,
#             'f': self.toggle_fullscreen,
#             's': self.take_screenshot,
#             'h': self.show_help
#         }
        
#         if chr(key) in key_actions:
#             key_actions[chr(key)]()

#     def cleanup(self):
#         """Cleanup resources"""
#         self.running = False
#         self.tracking_thread.join()
#         self.analysis_thread.join()
#         self.camera.release()
#         cv2.destroyAllWindows()
#         pygame.quit()
        
#         # Save analytics
#         self.save_analytics()

#     def save_analytics(self):
#         """Save analytics data"""
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         # Save CSV log
#         self.csv_logger.to_csv(
#             self.dirs['analytics'] / f'detection_log_{timestamp}.csv', 
#             index=False
#         )
        
#         # Save analytics summary
#         with open(self.dirs['analytics'] / f'analytics_{timestamp}.json', 'w') as f:
#             json.dump(self.analytics, f, indent=4)
        
#         # Save heatmap
#         plt.imsave(
#             self.dirs['analytics'] / f'heatmap_{timestamp}.png',
#             self.analytics['motion_heatmap'],
#             cmap='hot'
#         )

#     @staticmethod
#     def print_controls():
#         """Print system controls"""
#         controls = """
#         Ultimate Detection System Controls:
#         ---------------------------------
#         q: Quit
#         r: Toggle recording
#         m: Cycle through modes
#         t: Cycle through themes
#         a: Toggle analytics display
#         f: Toggle fullscreen
#         s: Take screenshot
#         h: Show help
#         ---------------------------------
#         """
#         print(controls)

# if __name__ == "__main__":
#     detector = UltimateDetectionSystem()
#     detector.run()