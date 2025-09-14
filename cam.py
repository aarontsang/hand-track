import cv2
import time
import platform
import subprocess

import mediapipe as mp

mp_hands = mp.solutions.hands

PINCH_ON = 0.1   # go to "pinched" when dist <= this (pause)
PINCH_OFF = 0.1000001  # go to "expanded" when dist >= this (play)
IDLE = 0.15
COOLDOWN_MS = 500 # avoid repeated triggers

def media_pause():
    try:
        subprocess.run(
            ["osascript", "-e", 'tell application "Spotify" to pause'],
            check=True, capture_output=True
        )
        return
    except Exception:
        pass

def media_play():
    try:
        subprocess.run(
            ["osascript", "-e", 'tell application "Spotify" to play'],
            check=True, capture_output=True
        )
        return
    except Exception:
        pass

def thumb_index_distance_norm(hand_landmarks):
    t = hand_landmarks.landmark[4]
    i = hand_landmarks.landmark[8]
    dx = t.x - i.x
    dy = t.y - i.y
    dz = t.z - i.z
    # Euclidean in normalized space
    return (dx*dx + dy*dy + dz*dz) ** 0.5

def main():
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    cap = cv2.VideoCapture(0)

    pinched = False
    last_action_ms = 0

    print("Pinch (thumb-index) to PAUSE, Expand to PLAY. Press 'q' to quit.")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        right_dist = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, cls in enumerate(results.multi_handedness):
                label = cls.classification[0].label
                if label == "Right":
                    rh = results.multi_hand_landmarks[idx]
                    right_dist = thumb_index_distance_norm(rh)

                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, rh, mp_hands.HAND_CONNECTIONS
                    )

                    t = rh.landmark[4] # thumb
                    i = rh.landmark[8] # index finger
                    h, w = frame.shape[:2]
                    tx, ty = int(t.x * w), int(t.y * h)
                    ix, iy = int(i.x * w), int(i.y * h)
                    cv2.circle(frame, (tx, ty), 7, (0, 255, 0), -1)
                    cv2.circle(frame, (ix, iy), 7, (0, 255, 0), -1)
                    cv2.line(frame, (tx, ty), (ix, iy), (0, 255, 0), 2)
                    break

        now_ms = int(time.time() * 1000)
        if right_dist is not None:
            cv2.putText(
                frame, f"thumb-index dist: {right_dist:.3f}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2
            )

            if not pinched and right_dist <= PINCH_ON:
                # transition to pinched => PAUSE
                if now_ms - last_action_ms >= COOLDOWN_MS:
                    media_pause()
                    pinched = True
                    last_action_ms = now_ms
            elif pinched and PINCH_OFF < right_dist < IDLE:
                # transition to expanded => PLAY
                if now_ms - last_action_ms >= COOLDOWN_MS:
                    media_play()
                    pinched = False
                    last_action_ms = now_ms
            elif right_dist >= IDLE:
                print("Idle state, waiting for pinch or expand...")
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
