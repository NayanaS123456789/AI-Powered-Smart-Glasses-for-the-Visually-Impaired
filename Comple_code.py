
import sys
import termios
import tty
import pygame
import cv2
import mediapipe as mp
import RPi.GPIO as GPIO
import pytesseract
import busio
import board

from gtts import gTTS
from picamera2 import Picamera2
from adafruit_vl53l1x import VL53L1X
from time import sleep
from pathlib import Path
import select


# GPIO pin configuration
BUTTON_1 = 17       # Object and Distance Detection
BUTTON_2 = 27       # Text Recognition
BUTTON_3 = 22       # Pose Estimation
BUTTON_QUIT = 23    # Quit program


# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_3, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_QUIT, GPIO.IN, pull_up_down=GPIO.PUD_UP)


# Initialize pygame for audio feedback
pygame.mixer.init()


# Global variables
picam2 = None
tof = None
pose = None
net = None
class_labels = []
initialized = False


def speak(text):
    """Convert text to speech and play the audio."""
    tts = gTTS(text=text, lang="en")
    tts.save("audio.mp3")

    pygame.mixer.music.load("audio.mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        sleep(0.1)


def initialize_system():
    """Initialize camera, distance sensor, pose model, and object detector."""
    global picam2, tof, pose, net, class_labels, initialized

    if initialized:
        return

    print("Initializing system...")

    # Initialize camera
    picam2 = Picamera2()

    config = picam2.create_still_configuration(
        main={"size": (300, 300)}
    )

    picam2.configure(config)
    picam2.start()
    sleep(2)

    # Initialize distance sensor
    i2c = busio.I2C(board.SCL, board.SDA)
    tof = VL53L1X(i2c)
    tof.start_ranging()

    # Initialize MediaPipe pose detection
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    )

    # Load MobileNet-SSD model
    model_path = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
    proto_path = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"

    net = cv2.dnn.readNetFromCaffe(
        proto_path,
        model_path
    )

    # Load class labels
    class_labels = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

    initialized = True

    print("System initialized.")


def take_photo():
    """Capture an image from the Raspberry Pi camera."""
    frame = picam2.capture_array()

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def detect_objects_with_mobilenet(image):
    """Detect objects using MobileNet-SSD."""
    blob = cv2.dnn.blobFromImage(
        image,
        0.007843,
        (300, 300),
        127.5
    )

    net.setInput(blob)
    detections = net.forward()

    detected_objects = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            class_id = int(detections[0, 0, i, 1])

            if 0 <= class_id < len(class_labels):
                detected_objects.append(
                    class_labels[class_id]
                )

    return detected_objects


def detect_pose(image):
    """Detect whether a human pose is visible."""
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = pose.process(image_bgr)

    if results.pose_landmarks:
        return "Pose detected"

    return "No pose detected"


def check_buttons():
    """Check which physical button is pressed."""
    if not GPIO.input(BUTTON_1):
        return BUTTON_1

    elif not GPIO.input(BUTTON_2):
        return BUTTON_2

    elif not GPIO.input(BUTTON_3):
        return BUTTON_3

    elif not GPIO.input(BUTTON_QUIT):
        return BUTTON_QUIT

    return None


def perform_task(task):
    """Perform the task selected by the user."""
    if not initialized:
        initialize_system()

    if task == BUTTON_1:
        # Object and distance detection
        frame = take_photo()

        detected_objects = detect_objects_with_mobilenet(frame)

        distance = tof.distance

        output_message = (
            f"Objects: "
            f"{', '.join(detected_objects) if detected_objects else 'None'}. "
            f"Distance: {distance} mm."
        )

    elif task == BUTTON_2:
        # Text recognition
        frame = take_photo()

        detected_text = pytesseract.image_to_string(
            frame,
            config="--psm 6"
        ).strip()

        output_message = (
            f"Recognized text: "
            f"'{detected_text if detected_text else 'No text detected'}'."
        )

    elif task == BUTTON_3:
        # Pose estimation
        frame = take_photo()

        pose_description = detect_pose(frame)

        output_message = f"Detected pose: {pose_description}."

    elif task == BUTTON_QUIT:
        print("Exiting program...")
        cleanup()
        raise SystemExit

    else:
        output_message = "Unknown command."

    print(output_message)
    speak(output_message)


def cleanup():
    """Release hardware and software resources."""
    global initialized

    if picam2 is not None:
        picam2.stop()

    if tof is not None:
        tof.stop_ranging()

    if pose is not None:
        pose.close()

    pygame.mixer.quit()
    GPIO.cleanup()

    initialized = False


def get_keypress():
    """Read a single keyboard character."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)

    finally:
        termios.tcsetattr(
            fd,
            termios.TCSADRAIN,
            old_settings
        )

    return ch


def main():
    print(
        "Press any button to initialize the system. "
        "Press 'q' to exit."
    )

    try:
        while True:
            task = check_buttons()

            if task is not None:
                perform_task(task)

                # Wait until the button is released
                while not GPIO.input(task):
                    sleep(0.05)

            # Non-blocking keyboard input
            if select.select([sys.stdin], [], [], 0)[0]:
                key = get_keypress()

                if key.lower() == "q":
                    print("Exiting program...")
                    break

            sleep(0.1)

    except KeyboardInterrupt:
        print("\nProgram interrupted.")

    finally:
        cleanup()


if __name__ == "__main__":
    main()
