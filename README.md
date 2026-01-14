**Crowd Vision Analyzer**

*System Description*

Crowd Vision Analyzer is an intelligent AI-powered video analytics system designed to automatically analyze people present in surveillance videos and live camera streams. The system uses YOLOv8 for accurate person detection, DeepSORT for multi-person tracking, and custom-trained deep learning models to identify whether individuals are wearing face masks, glasses, both, or none. It processes video in real time, assigns unique IDs to each person, and continuously monitors crowd behavior and compliance. The system provides accurate statistics, classifications, and visual overlays, making it a powerful tool for automated public monitoring and safety enforcement.

**what it do ?**

- Detect Total People in video or through web cam
- Detect total people wear mask
- Detect total people wear Glasses
- Detect Total People wear both
- Detect Total people wear nothing

**How To use It**

- Clone the github repo
- download the weights (link given in weights folder)
- create a virtual envirmoment
- Run the requirment.txt file
- clone the deepsort algo from "https://github.com/nwojke/deep_sort.git"
- replace the given track file with the track.py in deepsort algo
- run the code
- 
**Skill**

- Python
- computer vision
- object detection
- object tracking
