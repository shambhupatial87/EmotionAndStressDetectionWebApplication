import cv2
import numpy as np
import dlib
import time

# Initialize face detector and shape predictor for stress detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dependencies/shape_predictor_68_face_landmarks.dat")

def get_forehead(frame, landmarks):
    p8c = [landmarks[8][0], landmarks[8][1] - 2*(landmarks[8][1]-landmarks[29][1])]
    p27 = landmarks[27]
    forehead_width = 100
    forehead_height = 40
    forehead_offset = 20
    forehead_p1 = (p27[0] - forehead_width // 2, p8c[1] + forehead_offset)
    forehead_p2 = (p27[0] + forehead_width // 2, p8c[1] + forehead_offset + forehead_height)
    cv2.rectangle(frame, forehead_p1, forehead_p2, (0, 255, 0), 2)
    forehead = frame[forehead_p1[1]:forehead_p2[1], forehead_p1[0]:forehead_p2[0]]
    return forehead

def draw_landmarks(frame, landmarks):
    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

def calculate_stress_info(times, data_buffer, buffer_size, fps):
    stress = 0  
    processed = np.array(data_buffer)
    L = len(processed)
    
    if L > 10:
        even_times = np.linspace(times[0], times[-1], L) 
        interpolated = np.interp(even_times, times, processed) 
        interpolated = np.hamming(L) * interpolated 
        interpolated = interpolated - np.mean(interpolated) 
        raw = np.fft.rfft(interpolated) 
        phase = np.angle(raw)
        fft = np.abs(raw) 
        freqs = 60. * np.arange(L / 2 + 1) / L 
        freqs = freqs[1:] 
        idx = np.where((freqs > 10) & (freqs < 30)) 
        pruned = fft[idx] 
        if pruned.any(): 
            idx2 = np.argmax(pruned) 
            hri = freqs[idx2] 
            wait = (len(processed) - L) / fps
            for i in range(10):
                stress += hri
            if stress > 100: 
                stress /= 10
                stress += 10
    return stress

def detect_stress(frame, data_buffer, times, t0):
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0) 

    for rect in rects:
        landmarks = predictor(gray, rect) 
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()]) 

        draw_landmarks(frame, landmarks) 

        forehead = get_forehead(frame, landmarks)

        vals = np.mean(forehead)
        data_buffer.append(vals)
        times.append(time.time() - t0)
        L = len(data_buffer)
        if L > 10:  
            print('here')   
            stress2 = calculate_stress_info(times, data_buffer, len(data_buffer), 30)
            cv2.putText(frame, "Stress Level: {:.2f}".format(stress2), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(stress2)
            if L > 250:
                data_buffer = data_buffer[-250:]
                times = times[-250:]

    return frame
