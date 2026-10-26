

**An Edge AI-Based Assistive System for Real-Time Environmental Awareness and Accessibility**

## 📌 Project Overview

The AI-Powered Smart Glasses project aims to assist visually impaired individuals in understanding their surroundings through real-time computer vision, deep learning, and audio feedback.

The system integrates a camera, embedded computing hardware, and multiple AI-based modules to identify surrounding objects, estimate their distances, recognize human activities and body poses, extract text from the environment, and detect currency notes.

The processed information is converted into spoken feedback and delivered through a bone-conduction speaker, allowing users to receive environmental information while keeping their ears relatively unobstructed.

The project focuses on developing a **cost-effective, portable, and intelligent assistive technology solution** using Edge AI and embedded systems.

## 🎯 Project Objectives

- Develop an AI-powered wearable assistive system for visually impaired individuals.
- Enable real-time object detection and environmental awareness.
- Estimate the relative distance of objects using monocular depth estimation.
- Recognize human activities and body poses.
- Extract readable text using Optical Character Recognition (OCR).
- Identify currency notes using a deep-learning-based detection model.
- Provide accessible audio feedback through text-to-speech technology.
- Explore on-device AI processing for portable, low-latency operation.

## ✨ Key Features

### 1. Object Detection
Uses **YOLOv5** to identify objects in the camera's field of view.

- Detects supported objects in real time.
- Identifies object categories and their locations in the image.
- Provides object information for audio feedback.
- Supports environmental awareness and object identification.

### 2. Distance Estimation
Uses **MonoDepth2** for monocular depth estimation.

- Estimates relative depth from a single camera image.
- Helps distinguish objects at different apparent distances.
- Provides supplementary information for understanding the surroundings.

*Note: Monocular depth estimation generally produces relative depth unless calibrated to provide reliable metric distances. It should not be treated as a safety-certified obstacle-distance measurement.*

### 3. Human Activity and Pose Recognition
Integrates human activity recognition (HAR) and pose-estimation techniques to analyze human posture and movement.

- Identifies supported human poses or activities.
- Extracts useful information about people in the environment.
- Extends the system beyond basic object detection.

### 4. Text Recognition Using OCR
Uses **Tesseract OCR** to extract text from captured images.

- Recognizes text from suitable printed or displayed sources.
- Converts detected text into machine-readable output.
- Enables integration with text-to-speech for audio-based access to written information.

Recognition quality depends on lighting, image resolution, focus, text orientation, and font clarity.

### 5. Currency Detection
Uses a YOLO-based object detection model to identify supported currency notes.

- Detects currency notes in camera images.
- Provides an additional accessibility feature for everyday activities.
- Can be extended to support more currency denominations and currencies.

Currency recognition should be validated carefully before being relied upon for financial transactions.

### 6. Audio Feedback
Integrates Text-to-Speech (TTS) to communicate relevant AI predictions to the user.

- Converts recognized information into spoken output.
- Provides an accessible alternative to visual displays.
- Uses a bone-conduction speaker to deliver audio feedback.

### 7. Hardware-Based Controls
GPIO-connected physical switches provide a simple interface for activating selected functions.

The control system can be configured to trigger specific modules, such as:

- Object detection and distance estimation
- Text recognition
- Human activity or pose recognition

The exact switch mapping depends on the implemented hardware configuration.

## 🏗️ System Architecture

The system follows a camera-to-AI-to-audio processing pipeline.

```text
        Camera Module
              |
              v
       Image Acquisition
              |
              v
       Image Preprocessing
              |
              v
    +---------+----------+----------------+
    |         |          |                |
    v         v          v                v
 YOLOv5   MonoDepth2   HAR / Pose      Tesseract OCR
 Object   Depth        Recognition     Text Extraction
 Detection Estimation
    |
    +--------------------+
                         |
                  YOLO-Based Model
                  Currency Detection
                         |
                         v
                Result Processing
                         |
                         v
                 Text-to-Speech
                         |
                         v
              Bone-Conduction Speaker
                         |
                         v
                 Audio Feedback
```

The modules may operate independently or be triggered selectively according to the hardware controls and software implementation.

## 🔄 Working Principle

1. **Image acquisition:** A camera captures images of the user's surroundings.
2. **Preprocessing:** Captured frames are prepared for AI inference using suitable image-processing operations.
3. **AI inference:** The selected model analyzes the input for object detection, depth estimation, human activity or pose recognition, OCR, or currency detection.
4. **Result interpretation:** Relevant model outputs are processed into concise, meaningful information.
5. **Audio generation:** The processed information is converted into speech using TTS.
6. **Audio delivery:** A bone-conduction speaker communicates the generated feedback to the user.
7. **User interaction:** Physical GPIO switches allow the user to activate selected functions.

## 🛠️ Technologies and Tools

| Technology | Purpose |
|---|---|
| Python | Core programming and AI pipeline |
| OpenCV | Image acquisition and preprocessing |
| YOLOv5 | Object detection |
| MonoDepth2 | Monocular depth estimation |
| Human Activity Recognition | Activity and posture analysis |
| Tesseract OCR | Text extraction |
| YOLO-based currency model | Currency-note detection |
| Text-to-Speech (TTS) | Audio feedback |
| Raspberry Pi | Embedded computing platform |
| GPIO | Physical button and switch control |
| Bone-conduction speaker | Audio output |

*The table describes the intended system architecture. Adjust the entries to reflect the specific hardware, libraries, and models used in your final implementation.*

## 🔌 Hardware Components

The proposed hardware setup includes:

- **Embedded computing board:** Raspberry Pi or another compatible edge-computing platform.
- **Camera module:** Captures images of the surrounding environment.
- **Physical switches:** Allow users to activate specific functions.
- **Bone-conduction speaker:** Delivers spoken information.
- **Power supply:** Provides power to the computing board and connected peripherals.
- **Wearable frame:** Holds the camera and associated components in a suitable position.

Actual component models and wiring should be documented according to the hardware used in the prototype.

## 💻 Software Requirements

The software environment may include:

- Python 3
- OpenCV
- PyTorch and compatible YOLOv5 dependencies
- MonoDepth2 dependencies
- Tesseract OCR
- GPIO libraries compatible with the target hardware
- A compatible text-to-speech engine

The exact Python version and package versions depend on the selected embedded board and model implementations.

## 🚀 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with your actual GitHub repository URL and folder name.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux:

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Install only the dependencies compatible with your operating system, hardware, and selected AI models.

### Step 4: Install OCR Dependencies

Install the Tesseract OCR engine using the appropriate package manager for your operating system. Then install the Python interface if needed:

```bash
pip install pytesseract
```

The OCR engine itself must also be installed; installing the Python package alone is not sufficient.

### Step 5: Configure the AI Models

- Place model weights in the appropriate directories.
- Configure model paths in the relevant scripts.
- Verify that the selected models load correctly.
- Check camera access and hardware dependencies.
- Configure GPIO pins according to the circuit and switch mapping.

### Step 6: Run the Application

Run the project's actual entry-point script. For example, if the main script is named `main.py`:

```bash
python main.py
```

Replace `main.py` with the actual filename used in your repository.

## 📁 Suggested Repository Structure

```text
smart-glasses/
│
├── README.md
├── requirements.txt
├── main.py
│
├── models/
│   ├── object_detection/
│   ├── depth_estimation/
│   ├── activity_recognition/
│   └── currency_detection/
│
├── modules/
│   ├── image_preprocessing.py
│   ├── object_detection.py
│   ├── depth_estimation.py
│   ├── activity_recognition.py
│   ├── text_recognition.py
│   ├── currency_detection.py
│   └── audio_feedback.py
│
├── hardware/
│   └── gpio_controls.py
│
├── assets/
│   ├── architecture.png
│   └── demo_images/
│
└── docs/
    └── hardware_setup.md
```

This is a suggested structure, not a description of your current repository. Keep only the folders and scripts that exist in your implementation.

## 🧪 Testing and Evaluation

The system can be evaluated through module-wise and integrated testing.

| Module | Suggested evaluation |
|---|---|
| Object detection | Precision, recall, mAP, and inference time |
| Depth estimation | Depth consistency and error against reference measurements |
| Activity recognition | Accuracy, precision, recall, and F1-score |
| OCR | Character or word recognition error rate |
| Currency detection | Precision, recall, and performance across supported notes |
| Audio feedback | Response latency and intelligibility |
| Integrated system | End-to-end latency, reliability, and power consumption |

Testing should include different lighting conditions, object distances, camera angles, and environmental backgrounds. Report measured results only after conducting the corresponding experiments.

## 🌍 Applications

- Assistive environmental awareness
- Identification of everyday objects
- Access to printed text through audio
- Currency-note identification
- Human posture and activity awareness
- Wearable AI and accessibility research
- Embedded computer vision applications

## ⚠️ Limitations and Safety Considerations

- AI predictions may be inaccurate in unfamiliar or visually challenging environments.
- Depth estimation may not provide reliable real-world distances without calibration.
- OCR performance depends on image quality and text visibility.
- Currency detection is limited to the denominations and conditions supported by the trained model.
- Real-time performance depends on the hardware, model size, and optimization.
- The prototype is an assistive research system and should not be considered a replacement for a mobility cane, guide dog, or other established mobility aids.
- Users should not rely on the system as their sole means of detecting hazards.

## 🔮 Future Enhancements

- Optimize inference using lightweight models and hardware acceleration.
- Improve power efficiency for longer wearable operation.
- Add voice commands for hands-free interaction.
- Improve depth calibration and obstacle-awareness reliability.
- Expand OCR support to different languages.
- Extend currency recognition to additional denominations.
- Add GPS-based navigation and location assistance.
- Evaluate usability, response time, and reliability through structured user testing.

## 🎓 Project Focus

**Computer Vision | Edge AI | Deep Learning | Embedded Systems | Human Activity Recognition | Assistive Technology | Biomedical Engineering**

## 🤝 Contributions

Contributions and suggestions that improve accessibility, model performance, hardware integration, and usability are welcome.

If you would like to contribute, fork the repository, create a feature branch, and submit a pull request describing your changes.

## 📜 License

Add an appropriate open-source license to this repository if you intend to distribute the code. For example, a MIT License can be used when its terms fit your project and you have the right to license all included code and assets.

## ⚕️ Disclaimer

This project is developed for educational and research purposes to explore AI-powered assistive technology. It is not a certified medical device or a safety-critical navigation system. Its outputs may contain errors and should not be used as the sole basis for navigation or other safety-critical decisions.
