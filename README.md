# Crowd Vision Analyzer

## System Description

**Crowd Vision Analyzer** is an AI-powered video analytics system that automatically detects and tracks people in surveillance videos and live camera streams.

The system uses:

* **YOLOv8** for accurate person detection
* **DeepSORT** for multi-person tracking
* Custom deep learning models to classify if individuals are wearing **face masks, glasses, both, or none**

It operates in real-time, assigns unique IDs to each person, and continuously monitors crowd behavior and compliance. The system provides **accurate statistics, classifications, and visual overlays**, making it ideal for automated public monitoring and safety enforcement.

---

## Features

* Detect total people in video or through webcam
* Detect total people wearing a **mask**
* Detect total people wearing **glasses**
* Detect total people wearing **both mask and glasses**
* Detect total people wearing **nothing**

---

## Output Example

![Crowd Vision Detection Demo](demo.gif)


## How To Use

### 1. Clone the repository

```bash
git clone <repo-link>
cd Crowd-Vision-Analyzer
```

### 2. Download the weights

* Link is provided in the `weights` folder

### 3. Create a virtual environment

```bash
python -m venv venv
# Activate virtual environment
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Clone DeepSORT repository

```bash
git clone https://github.com/nwojke/deep_sort.git
```

### 6. Replace `track.py`

* Copy the provided `track.py` from this repo into the cloned DeepSORT folder

### 7. Run the application

```bash
python main.py
```

---

## Skills & Technologies

* **Python**
* **Computer Vision**
* **Object Detection (YOLOv8)**
* **Object Tracking (DeepSORT)**

---


