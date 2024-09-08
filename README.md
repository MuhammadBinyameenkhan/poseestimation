**Marmoset Pose Estimation and Instance Segmentation Pipeline**
This repository demonstrates how to fine-tune Ultralytics YOLOv8 for pose estimation using the marmoset dataset and how to generate masks using Segment Anything Model (SAM) or SAM2 for combining with YOLOv8 for instance segmentation.
**Requirements**
Google Colab
Python 3.x
Ultralytics YOLOv8
Segment Anything Model (SAM) / SAM2
Marmoset Dataset from DeepLabCut Benchmark
Steps to Set Up and Run
1. Download the Marmoset Dataset
Download the marmoset dataset from DeepLabCut Benchmark.

Once the dataset is downloaded, upload it to Google Colab or mount your Google Drive for easy access.
**Mount Google Drive in Colab:**
from google.colab import drive
drive.mount('/content/drive')
**Copy the dataset to your Colab workspace:**
!cp /content/drive/MyDrive/marmoset_dataset.zip /content/
!unzip /content/marmoset_dataset.zip -d /content/marmoset_data
 **Install Required Libraries**
Install Ultralytics YOLOv8 and SAM or SAM2:
!pip install ultralytics
!pip install git+https://github.com/facebookresearch/segment-anything.git  # For SAM
 **Fine-Tune YOLOv8 for Pose Estimation**
Set up the configuration for the pose estimation model. Create a YAML file for YOLOv8 to fine-tune using the marmoset dataset.

Create marmoset_pose.yaml:
# marmoset_pose.yaml
path: /content/marmoset_data  # Dataset root path

train: /content/marmoset_data/images/train  # Path to training images
val: /content/marmoset_data/images/val  # Path to validation images

nc: 15  # Number of keypoints (adjust for your dataset)
names: ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee']  # Keypoint names
**Train the YOLOv8 Model:**
from ultralytics import YOLO

# Load YOLOv8 model for pose estimation
model = YOLO('yolov8n-pose.pt')

# Train the model
model.train(data='/content/marmoset_pose.yaml', epochs=50, imgsz=640)
 **Generate Masks using SAM or SAM2**
Use SAM or SAM2 to generate masks from each frame of the marmoset dataset.
from segment_anything import SamPredictor
from segment_anything import sam_model_registry

# Load the SAM model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

# Run SAM on an image
image_path = '/content/marmoset_data/images/train/sample_image.jpg'
image = cv2.imread(image_path)
predictor.set_image(image)

masks = predictor.predict()
**Combine Mask with YOLO Bounding Box for Instance Segmentation**
After generating the masks from SAM, combine them with YOLOv8 bounding boxes to perform instance segmentation.
from ultralytics import YOLO

# Load the YOLOv8 model (instance segmentation)
yolo_model = YOLO('yolov8n-seg.pt')

# Inference on a test image to get bounding boxes
results = yolo_model('path/to/image.jpg')

# Combine YOLO bounding boxes and SAM-generated masks
for result, mask in zip(results.xyxy, masks):
    # Extract bounding boxes and masks
    bbox = result.xyxy
    instance_mask = mask  # From SAM
    
    # Combine as needed (visualization, metrics, etc.)
    
**Summary of Steps**
Download the marmoset dataset.
Fine-tune YOLOv8 for pose estimation using marmoset_pose.yaml.
Use SAM or SAM2 to generate masks for each frame.
Combine YOLOv8 bounding boxes with SAM-generated masks for instance segmentation.
**Important Links**
Marmoset Dataset: DeepLabCut Benchmark
Ultralytics YOLOv8: Ultralytics Documentation
SAM: SAM Documentation
SAM2: SAM2 Documentation
