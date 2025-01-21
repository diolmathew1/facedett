# Employee Biometric Attendance System

## Overview
This project implements a **biometric attendance system** using face detection and recognition. It combines computer vision techniques with machine learning to automate employee attendance. The system includes features for employee registration, training a machine learning model, and live attendance monitoring.

## Features
- **Employee Registration**: Allows employees to register with personal details and captures facial images for training.
- **Model Training**: Trains a Support Vector Machine (SVM) model to recognize employee faces.
- **Attendance Marking**: Uses the trained model to detect and identify employees in real-time.
- **Data Storage**: Maintains employee records and attendance logs in CSV files.

## Installation
### Prerequisites
- Python 3.8 or higher
- Libraries: `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tkinter`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/employee-attendance.git
   ```
2. Navigate to the project directory:
   ```bash
   cd employee-attendance
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Employee Registration
Run the application, fill in the employee details, and capture face images for model training.

### 2. Model Training
Train the SVM model on the collected face images.

### 3. Live Attendance
Start the live attendance mode to mark attendance using real-time face recognition.

## Directory Structure
```
employee-attendance/
├── 01-research_report/
│   └── research_report.pdf
├── 02-code/
│   ├── main.py
│   ├── utils/
│   │   └── helper_functions.py
│   └── requirements.txt
└── README.md
```