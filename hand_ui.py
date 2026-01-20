import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import random  # For particle randomization

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Neon colors
NEON_BLUE = (255, 120, 0)
NEON_PURPLE = (255, 0, 180)
NEON_CYAN = (255, 255, 0)
NEON_PINK = (180, 0, 255)
WHITE = (255, 255, 255)
ACCENT_GREEN = (100, 255, 150)
RED = (0, 0, 255)

# ---------- Particle Class ----------
class Particle:
    def __init__(self, pos, vel, color, lifetime=30):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.color = color
        self.lifetime = lifetime
        self.age = 0

    def update(self):
        self.pos += self.vel
        self.vel *= 0.98  # Damping
        self.age += 1

    def draw(self, img):
        if self.age < self.lifetime:
            alpha = 1 - (self.age / self.lifetime)
            cv2.circle(img, tuple(self.pos.astype(int)), 2, self.color, -1)
            # Simple glow effect
            cv2.circle(img, tuple(self.pos.astype(int)), 5, self.color, 1)

# ---------- Modern UI Class ----------
class ModernUI:
    def __init__(self):
        self.frame_count = 0
        self.gesture_state = "IDLE"
        self.last_gesture = "IDLE"
        self.last_pos = None   # for freehand drawing
        self.canvas = None  # Persistent drawing canvas
        self.particles = []  # List of particles
        self.particle_timer = 0
        self.shape_mode = 0  # 0: Freehand, 1: Circle, 2: Square, 3: Triangle
        self.shapes = ["Freehand", "Circle", "Square", "Triangle"]

    # Glow effect
    def draw_glow(self, img, center, radius, color, intensity=0.3):
        overlay = np.zeros_like(img, dtype=np.uint8)
        for i in range(5):
            alpha = intensity * (1 - i/5)
            cv2.circle(overlay, center, radius + i*8, color, -1)
            cv2.addWeighted(img, 1, overlay, alpha, 0, img)
            overlay.fill(0)

    # Freehand line drawing on canvas (thinner)
    def draw_line(self, start, end, color=NEON_CYAN):
        cv2.line(self.canvas, start, end, color, 2)  # Thinner line
        self.draw_glow(self.canvas, end, 3, color, 0.3)  # Smaller glow

    # Draw shapes on canvas
    def draw_shape(self, center, shape, size=50, color=NEON_PINK):
        if shape == "Circle":
            cv2.circle(self.canvas, center, size, color, 2)
        elif shape == "Square":
            x, y = center
            cv2.rectangle(self.canvas, (x - size, y - size), (x + size, y + size), color, 2)
        elif shape == "Triangle":
            x, y = center
            points = np.array([[x, y - size], [x - size, y + size], [x + size, y + size]], np.int32)
            cv2.polylines(self.canvas, [points], True, color, 2)

    # Emit particles
    def emit_particles(self, center, num_particles=20):
        for _ in range(num_particles):
            angle = random.uniform(0, 2*np.pi)
            speed = random.uniform(1, 5)
            vel = [speed * np.cos(angle), speed * np.sin(angle)]
            color = random.choice([NEON_BLUE, NEON_PURPLE, NEON_CYAN, NEON_PINK])
            self.particles.append(Particle(center, vel, color))

    # Update particles
    def update_particles(self, img):
        for p in self.particles[:]:
            p.update()
            if p.age >= p.lifetime:
                self.particles.remove(p)
            else:
                p.draw(img)

    # Status panel
    def draw_status_panel(self, img, gesture, fps, shape):
        x, y, w, h = 10, 10, 350, 100
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.putText(img, f'Gesture: {gesture}', (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        cv2.putText(img, f'Shape Mode: {shape}', (x+10, y+55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT_GREEN, 1)
        cv2.putText(img, f'FPS: {fps}', (x+10, y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT_GREEN, 1)

# ---------- Initialize ----------
ui = ModernUI()
cap = cv2.VideoCapture(0)
prev_time = 0

print("Starting Advanced Hand Tracking AR UI...")
print("Gestures:")
print("  - Open Hand: Draw alphabets/shapes with any/index finger + Particles on open")
print("  - Pinch: Cycle shape modes (Freehand -> Circle -> Square -> Triangle)")
print("  - Fist: Locked / No action")
print("Press ESC to exit")

# ---------- Main Loop ----------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if ui.canvas is None:
        ui.canvas = np.zeros_like(frame)  # Initialize canvas

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # FPS
    current_time = cv2.getTickCount()
    fps = int(cv2.getTickFrequency() / (current_time - prev_time)) if prev_time > 0 else 0
    prev_time = current_time

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = [(int(l.x * w), int(l.y * h)) for l in hand_landmarks.landmark]

            # Hand skeleton
            for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
                cv2.line(frame, lm[start_idx], lm[end_idx], (100, 100, 100), 1)

            palm = lm[9]
            index_tip = lm[8]

            # Distance metrics for gestures
            pinch_dist = np.linalg.norm(np.array(lm[4]) - np.array(lm[8]))
            avg_dist = np.mean([np.linalg.norm(np.array(lm[i]) - np.array(palm)) for i in [4,8,12,16,20]])

            # Gesture detection
            if avg_dist > 70:  # Open Hand => Draw alphabets/shapes + Particles on transition
                ui.gesture_state = "OPEN HAND"
                if ui.last_gesture != "OPEN HAND":  # Transition to open
                    ui.emit_particles(palm)
                if ui.shapes[ui.shape_mode] == "Freehand":
                    # Draw with any finger (use index as example, but can adapt for others)
                    if ui.last_pos is not None:
                        ui.draw_line(ui.last_pos, index_tip)
                    ui.last_pos = index_tip
                else:
                    # Draw shape at index tip
                    ui.draw_shape(index_tip, ui.shapes[ui.shape_mode])
                    ui.last_pos = None
            elif pinch_dist < 40:  # Pinch => Cycle shapes
                ui.gesture_state = "PINCH"
                if ui.last_gesture != "PINCH":  # On transition
                    ui.shape_mode = (ui.shape_mode + 1) % len(ui.shapes)
                ui.last_pos = None
            else:  # Fist
                ui.gesture_state = "FIST"
                ui.last_pos = None

            ui.last_gesture = ui.gesture_state
    else:
        ui.gesture_state = "NO HAND"
        ui.last_pos = None
        ui.last_gesture = "NO HAND"

    # Update and draw particles
    ui.update_particles(frame)

    # Overlay canvas on frame
    cv2.addWeighted(frame, 1, ui.canvas, 0.8, 0, frame)

    # Draw status panel
    ui.draw_status_panel(frame, ui.gesture_state, fps, ui.shapes[ui.shape_mode])

    ui.frame_count += 1
    cv2.imshow('Advanced Hand Tracking AR', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
print("Application closed.")