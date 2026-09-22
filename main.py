"""
Hand Detection & Finger Tracking
=================================
A beginner-friendly Computer Vision application using OpenCV and MediaPipe.

Features:
- Detects up to 2 hands in real time via your webcam.
- Identifies whether each detected hand is Left or Right.
- Tracks and displays all 21 hand landmarks and connections.
- Accurately counts how many fingers are currently raised.
- Displays a clean real-time Heads-Up Display (HUD) overlay.
- Shows real-time frames per second (FPS).
- Exits cleanly when pressing 'q' or 'Q'.
"""

import sys
import time
import cv2
import mediapipe as mp


def count_raised_fingers(landmarks, hand_label):
    """
    Determines how many fingers are raised for a given hand.

    Parameters:
        landmarks: MediaPipe normalized hand landmarks list (21 points).
        hand_label (str): "Left" or "Right" hand classification.

    Returns:
        int: Total number of raised fingers (0 to 5).
        dict: Status of each individual finger (True if raised, False otherwise).

    How this works:
    ---------------
    In computer vision, coordinate (0, 0) is at the TOP-LEFT corner of the image.
    - The Y-axis increases downwards.
    - The X-axis increases to the right.

    1. Four Fingers (Index, Middle, Ring, Pinky):
       - If the fingertip Y-coordinate is LESS than the corresponding PIP joint Y-coordinate,
         the finger is pointing upwards (raised).
       - Landmark IDs:
         - Index: Tip 8, PIP joint 6
         - Middle: Tip 12, PIP joint 10
         - Ring: Tip 16, PIP joint 14
         - Pinky: Tip 20, PIP joint 18

    2. Thumb:
       - The thumb moves laterally (sideways) across the palm rather than vertically.
       - In a mirrored selfie webcam view:
         - For a RIGHT hand (facing camera): Thumb extends outwards to the LEFT (tip.x < ip.x).
         - For a LEFT hand (facing camera): Thumb extends outwards to the RIGHT (tip.x > ip.x).
       - Landmark IDs:
         - Thumb Tip: 4
         - Thumb IP joint: 3
    """
    finger_status = {
        "Thumb": False,
        "Index": False,
        "Middle": False,
        "Ring": False,
        "Pinky": False,
    }

    # 1. Check Thumb (horizontal movement depending on handedness in mirror view)
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    if hand_label == "Right":
        # For Right hand in mirror view, extended thumb points to screen-left (smaller X)
        finger_status["Thumb"] = thumb_tip.x < thumb_ip.x
    else:
        # For Left hand in mirror view, extended thumb points to screen-right (larger X)
        finger_status["Thumb"] = thumb_tip.x > thumb_ip.x

    # 2. Check the remaining 4 fingers (vertical comparison: tip vs PIP joint)
    finger_landmarks = [
        ("Index", 8, 6),
        ("Middle", 12, 10),
        ("Ring", 16, 14),
        ("Pinky", 20, 18),
    ]

    for name, tip_idx, pip_idx in finger_landmarks:
        # Lower Y value in screen coordinates means higher up on the screen
        finger_status[name] = landmarks[tip_idx].y < landmarks[pip_idx].y

    total_raised = sum(1 for is_raised in finger_status.values() if is_raised)
    return total_raised, finger_status


def draw_hud(frame, detected_hands_info, fps=0):
    """
    Renders an informative, semi-transparent Heads-Up Display (HUD) overlay
    directly on the webcam frame.

    Parameters:
        frame: OpenCV BGR image frame.
        detected_hands_info: List of dicts containing hand metadata:
                             [{"label": "Right", "fingers": 3}, ...]
        fps (float): Real-time frames per second calculation.
    """
    h, w, _ = frame.shape
    num_hands = len(detected_hands_info)

    # Calculate dynamic height for the HUD box based on detected hands
    hud_width = 300
    base_height = 80
    per_hand_height = 55
    hud_height = base_height + (num_hands * per_hand_height if num_hands > 0 else 30)

    # Create a semi-transparent dark background card for clear readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (15 + hud_width, 15 + hud_height), (20, 20, 20), -1)
    # Blend overlay with original frame (65% dark card, 35% original background)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Draw card border
    cv2.rectangle(frame, (15, 15), (15 + hud_width, 15 + hud_height), (0, 200, 255), 2)

    # 1. Header Title
    cv2.putText(
        frame,
        "HAND DETECTION",
        (30, 45),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )

    # 2. Total Hands Detected & Real-time FPS
    fps_display = int(round(fps))
    cv2.putText(
        frame,
        f"Hands: {num_hands} | FPS: {fps_display}",
        (30, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 3. Individual Hand Details
    if num_hands == 0:
        cv2.putText(
            frame,
            "Show your hand to the camera...",
            (30, 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
    else:
        y_cursor = 105
        for i, info in enumerate(detected_hands_info):
            label = info["label"]
            fingers = info["fingers"]

            # Accent color: Cyan for Right hand, Orange-Yellow for Left hand
            color = (255, 200, 0) if label == "Right" else (0, 215, 255)

            cv2.putText(
                frame,
                f"Hand: {label}",
                (30, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Fingers Raised: {fingers}",
                (30, y_cursor + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y_cursor += per_hand_height

    # 4. Quit prompt at the bottom left of the screen
    cv2.putText(
        frame,
        "Press 'Q' to quit",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (100, 255, 100),
        2,
        cv2.LINE_AA,
    )


def draw_hand_badge(frame, hand_landmarks, label, fingers_count):
    """
    Draws a quick-glance badge floating right near the user's wrist on the webcam feed.
    """
    h, w, _ = frame.shape
    wrist = hand_landmarks.landmark[0]
    wrist_x = int(wrist.x * w)
    wrist_y = int(wrist.y * h)

    badge_text = f"{label}: {fingers_count}"
    badge_pos = (max(10, wrist_x - 40), min(h - 10, wrist_y + 30))

    # Badge background
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(
        frame,
        (badge_pos[0] - 5, badge_pos[1] - th - 5),
        (badge_pos[0] + tw + 5, badge_pos[1] + 5),
        (30, 30, 30),
        -1,
    )
    cv2.rectangle(
        frame,
        (badge_pos[0] - 5, badge_pos[1] - th - 5),
        (badge_pos[0] + tw + 5, badge_pos[1] + 5),
        (0, 255, 0),
        1,
    )

    # Text
    cv2.putText(
        frame,
        badge_text,
        badge_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    print("=" * 60)
    print(" Hand Detection & Finger Tracking (OpenCV + MediaPipe)")
    print("=" * 60)
    print("Initializing webcam...")

    # 1. Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[WARNING] Could not open camera at index 0. Trying camera index 1...")
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("\n[ERROR] Unable to access any webcam.")
        print("Please check that:")
        print("  1. Your webcam is properly connected.")
        print("  2. Other applications (Teams, Zoom, browser) are not using it.")
        print("  3. Windows camera permissions are granted for Python/Apps.")
        sys.exit(1)

    # Set camera resolution (optional: 640x480 for smooth real-time performance)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 2. Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Custom styling for landmarks and connections
    landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3)
    connection_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)

    hands = mp_hands.Hands(
        static_image_mode=False,        # Video stream mode for optimal tracking
        max_num_hands=2,                # Detect up to 2 hands
        min_detection_confidence=0.7,   # High confidence threshold for detection
        min_tracking_confidence=0.5,    # Smooth tracking threshold
    )

    print("\nWebcam successfully opened!")
    print("  - Detecting up to 2 hands in real time.")
    print("  - Tracking 21 3D hand landmarks.")
    print("  - Press 'Q' inside the video window to quit.\n")

    prev_time = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("[WARNING] Failed to grab frame from webcam. Retrying...")
                continue

            # Calculate real-time FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time

            # Flip frame horizontally for a natural selfie/mirror experience
            frame = cv2.flip(frame, 1)

            # Convert BGR (OpenCV default) to RGB (MediaPipe requirement)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame to detect hands
            results = hands.process(rgb_frame)

            detected_hands_info = []

            # If hands are detected
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # In mirror/selfie mode, MediaPipe's handedness label directly matches the user's hand
                    hand_label = handedness.classification[0].label  # "Left" or "Right"

                    # 1. Count raised fingers
                    fingers_count, _ = count_raised_fingers(
                        hand_landmarks.landmark, hand_label
                    )

                    # 2. Store information for the HUD
                    detected_hands_info.append({
                        "label": hand_label,
                        "fingers": fingers_count,
                    })

                    # 3. Draw the 21 landmarks and skeletal connections
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=landmark_style,
                        connection_drawing_spec=connection_style,
                    )

                    # 4. Draw quick-glance badge floating next to wrist
                    draw_hand_badge(frame, hand_landmarks, hand_label, fingers_count)

            # Draw the Heads-Up Display (HUD) overlay
            draw_hud(frame, detected_hands_info, fps)

            # Display the resulting video frame
            cv2.imshow("Hand Detection & Finger Tracking", frame)

            # Check if user pressed 'q' or 'Q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\nQuit key ('Q') pressed. Exiting cleanly...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    finally:
        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("Webcam and OpenCV windows released successfully.")


if __name__ == "__main__":
    main()
