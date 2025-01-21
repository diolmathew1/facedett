# Employee Biometric Attendance System

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
   git clone https://github.com/diolmathew1/facedett.git
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
