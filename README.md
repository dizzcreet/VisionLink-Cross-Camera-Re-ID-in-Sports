# VisionLink – Cross-Camera Re-ID in Sports

**VisionLink** is a machine learning–based computer vision system for re-identifying players across multiple camera views in sports footage. It ensures that the same player retains a consistent ID, even when switching between tactical and broadcast angles.

---

## 🧠 Project Description

VisionLink uses a fine-tuned YOLOv11 model to detect players and spatial features like bounding box centers to match identities across views. It serves as a baseline pipeline for real-world sports analytics tasks such as multi-angle player tracking, and is modular enough to extend with appearance-based or temporal techniques.

---

## 📁 Project Structure

player_reid_project/
├── models/
│ └── best.pt # YOLOv11 model (add manually)
├── videos/
│ ├── broadcast.mp4 # Broadcast view (add manually)
│ └── tacticam.mp4 # Tactical view (add manually)
├── utils/
│ ├── detection.py # YOLO-based detection
│ ├── feature_extraction.py # Extract bounding box features
│ ├── matching.py # Match players across views
│ └── visualization.py # Draw bounding boxes & IDs
├── main.py # Main pipeline script
├── requirements.txt # Dependencies
├── README.md # You're reading this!
└── report.md # Project approach and challenges


---

## 🛠️ Setup Instructions

### 1. Clone or Download
```bash
git clone https://github.com/yourusername/visionlink-reid.git
cd visionlink-reid
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Add Required Files
Manually place the following files in your project:

models/best.pt: [Download the model here](https://drive.google.com/drive/folders/1Nx6H_n0UUI6L-6i8WknXd4Cv2c3VjZTP)

videos/broadcast.mp4 and videos/tacticam.mp4: https://drive.google.com/drive/folders/1Nx6H_n0UUI6L-6i8WknXd4Cv2c3VjZTP
▶️ How to Run
bash
Copy
Edit
python main.py
Two video windows will open: one for broadcast view and one for tacticam

Players will be detected and tracked with consistent IDs

Press Q to quit at any time

💾 To Save Output
In main.py, set:

python
Copy
Edit
save_output = True
This will create output_broadcast.mp4 and output_tacticam.mp4.
```
💡 Techniques Used
Fine-tuned YOLOv11 for player detection

Spatial feature extraction (bounding box centers)

Euclidean distance for cross-camera matching

Real-time frame-by-frame processing

Modular pipeline for easy enhancement

🚀 Future Enhancements
Add deep visual embeddings (e.g., ResNet features)

Use temporal tracking (e.g., Deep SORT or ByteTrack)

Incorporate jersey color histograms for appearance-based matching


📄 Report
Refer to report.md for a detailed overview of:

Approach and methodology

Techniques attempted

Challenges encountered

What remains or could be improved
