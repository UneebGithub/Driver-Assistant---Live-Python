import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Dashboard setup
plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(131, projection='3d') # Left: 3D Depth View
ax2 = fig.add_subplot(132)                   # Center: Segmentation
ax3 = fig.add_subplot(133)                   # Right: Top-view Tracking

model = YOLO('yolov8n-seg.pt') 

# --- CAMERA CALIBRATION (Estimation) ---
# Maan lijiye agar person 500 pixels door hai toh wo car ke paas hai (0m)
# Aur agar 100 pixels door hai toh wo kafi door hai.
# Formula: Meters = (Pixel_Distance) * Scale_Factor
PIXEL_TO_METER_SCALE = 0.02  # Ye factor aapke camera setup ke hisaab se adjust hoga

def estimate_distance(y_coord, frame_height):
    # Camera perspective mein niche wali cheezein paas hoti hain
    # Distance = frame ki height se kitna upar hai
    dist_px = frame_height - y_coord
    return dist_px * PIXEL_TO_METER_SCALE

cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    h_frame, w_frame, _ = frame.shape

    results = model.track(frame, persist=True, classes=[0, 2]) # Person & Car

    ax1.cla(); ax2.cla(); ax3.cla()

    # Draw "Danger Zone" (3-Meter Line) on frame
    # 3m / 0.02 = 150 pixels from bottom
    safety_line_y = h_frame - 150 
    cv2.line(frame, (0, safety_line_y), (w_frame, safety_line_y), (0, 0, 255), 2)
    cv2.putText(frame, "3-METER SAFETY BARRIER", (10, safety_line_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if results[0].boxes.id is not None:
        seg_img = results[0].plot(labels=False, boxes=True)
        
        for box in results[0].boxes:
            x_c, y_c, w, h = box.xywh[0].cpu().numpy()
            dist_m = estimate_distance(y_c, h_frame)
            
            # --- RANGE LOGIC ---
            if dist_m <= 3.0:
                alert_color = 'red'
                label_color = (0, 0, 255) # Red for danger
                # Alert on screen
                cv2.circle(frame, (int(x_c), int(y_c)), 10, (0, 0, 255), -1)
            else:
                alert_color = 'lime'
                label_color = (0, 255, 0) # Green for safe

            # Display distance on video
            cv2.putText(frame, f"{dist_m:.1f}m", (int(x_c - w/2), int(y_c - h/2) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

            # --- PLOTTING GRAPHS ---
            # 3D Graph (Left)
            ax1.scatter(x_c, dist_m, (h_frame - y_c)/10, c=alert_color, s=100)
            
            # Top-View (Right)
            ax3.scatter(x_c, dist_m, c=alert_color, s=80, edgecolors='white')
            ax3.text(x_c, dist_m, f"{dist_m:.1f}m", color='white', fontsize=8)

        ax2.imshow(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB))

    # Graph Formatting
    ax1.set_title("3D Depth (Meters)")
    ax1.set_zlim(0, 10)
    ax3.set_title("Top View (3m Proximity)")
    ax3.set_ylim(0, 10) # 10 meter total range
    ax3.axhline(y=3, color='r', linestyle='--', label='3m Limit') # Danger line on graph

    plt.pause(0.01)
    cv2.imshow("Driver Assistant - Live", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
plt.close()