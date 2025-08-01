import cv2
import time
import mediapipe as mp
from pygame import mixer


mixer.init()
beep_sound = "beep.wav"
alarm_sound = "beep.wav"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

def get_eye_ratio(landmarks, eye_points, image_w, image_h):
    points = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in eye_points]
    hor_line = cv2.norm((points[0][0] - points[3][0], points[0][1] - points[3][1]))
    ver_line = cv2.norm((points[1][0] - points[5][0], points[1][1] - points[5][1]))
    ratio = ver_line / hor_line
    return ratio

eye_closed_time = None
beep_played = False
alarm_played = False

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
          
            for eye_id in LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS:
                x = int(face_landmarks.landmark[eye_id].x * w)
                y = int(face_landmarks.landmark[eye_id].y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            left_ratio = get_eye_ratio(face_landmarks.landmark, LEFT_EYE_LANDMARKS, w, h)
            right_ratio = get_eye_ratio(face_landmarks.landmark, RIGHT_EYE_LANDMARKS, w, h)
            avg_ratio = (left_ratio + right_ratio) / 2

            if avg_ratio < 0.08: 
                if eye_closed_time is None:
                    eye_closed_time = time.time()

                elapsed = time.time() - eye_closed_time

                if 5 <= elapsed < 15 and not beep_played:
                    mixer.music.load(beep_sound)
                    mixer.music.play()
                    beep_played = True

                elif elapsed >= 15 and not alarm_played:
                    mixer.music.load(alarm_sound)
                    mixer.music.play()
                    alarm_played = True

                cv2.putText(frame, f"Eyes Closed ({int(elapsed)}s)", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            else:
                if eye_closed_time is not None:
                    if time.time() - eye_closed_time > 4:
                        mixer.music.stop()
                        beep_played = False
                        alarm_played = False
                        eye_closed_time = None
                    else:
                        cv2.putText(frame, "Eyes Opening...", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Eyes Open", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "Face Not Detected", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Eye Alert System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
