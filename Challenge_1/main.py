import os
import cv2
import numpy as np
import json
import uuid
import shutil
from typing import List, Dict
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Tuple


app = FastAPI()

UPLOAD_DIR = "images"
circle_data_store = {}  # Dictionary to store circle data

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Load ground truth data from JSON file
GROUND_TRUTH_FILE = "coin-dataset/_annotations.coco.json"
try:
    with open(GROUND_TRUTH_FILE, "r") as f:
        ground_truth_data = json.load(f)
except FileNotFoundError:
    # If the ground truth file is not found, set an empty dictionary
    ground_truth_data = {}

def load_ground_truth(image_path: str) -> List[Dict]:
    """
    Load ground truth data for a given image.

    Parameters:
        image_path (str): Path to the image.

    Returns:
        List[Dict]: Ground truth circles data for the image.
    """
    image_name = os.path.basename(image_path)
    return ground_truth_data.get(image_name, [])

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Endpoint to upload an image file.

    Parameters:
        file (UploadFile): The image file to upload.

    Returns:
        Dict: Information about the uploaded file.
    """
    try:
        file_location = f"{UPLOAD_DIR}/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"info": "File uploaded successfully", "file_path": file_location}
    except Exception as e:
        return {"error": str(e)}

@app.get("/circles")
def get_circles(image_path: str):
    """
    Endpoint to detect circles in an uploaded image.

    Parameters:
        image_path (str): Path to the uploaded image.

    Returns:
        JSONResponse: Detected circles data.
    """
    try:
        circles = detect_circles(image_path)
        circle_data = []
        if circles is not None:
            for circle in circles:
                x, y, r = map(int, circle)
                id = str(uuid.uuid4())
                circle_info = {"id": id, "bounding_box": [x-r, y-r, x+r, y+r], "centroid": [x, y], "radius": r}
                circle_data_store[id] = circle_info
                circle_data.append(circle_info)
        return JSONResponse(content={"circles": circle_data})
    except Exception as e:
        return {"error": str(e)}

@app.get("/circle/{circle_id}")
def get_circle(circle_id: str):
    """
    Endpoint to retrieve information about a specific circle.

    Parameters:
        circle_id (str): Unique identifier of the circle.

    Returns:
        JSONResponse: Information about the circle.
    """
    try:
        if circle_id in circle_data_store:
            return JSONResponse(content=circle_data_store[circle_id])
        else:
            return JSONResponse(content={"error": "Circle not found"}, status_code=404)
    except Exception as e:
        return {"error": str(e)}

def detect_circles(image_path: str) -> List[Tuple[int]]:
    """
    Detect circles in an image using Hough Circle Transform.

    Parameters:
        image_path (str): Path to the image.

    Returns:
        List[Tuple[int]]: Detected circles (x, y, radius).
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=50)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return circles.tolist()
        else:
            return []
    except Exception as e:
        raise ValueError(f"Error detecting circles: {e}")

@app.get("/evaluate_model")
def get_evaluation(image_path: str):
    """
    Endpoint to evaluate the circle detection model.

    Parameters:
        image_path (str): Path to the uploaded image.

    Returns:
        JSONResponse: Evaluation results (precision, recall, F1-score).
    """
    try:
        evaluation_results = evaluate_model(image_path)
        return JSONResponse(content=evaluation_results)
    except Exception as e:
        return {"error": str(e)}

def evaluate_model(image_path: str) -> Dict[str, float]:
    """
    Evaluate the circle detection model using ground truth data.

    Parameters:
        image_path (str): Path to the uploaded image.

    Returns:
        Dict[str, float]: Evaluation results (precision, recall, F1-score).
    """
    try:
        detected_circles = detect_circles(image_path)
        if not detected_circles:
            raise ValueError("No circles detected")

        ground_truth_circles = load_ground_truth(image_path)
        if not ground_truth_circles:
            raise ValueError("Ground truth data not available")

        # Initialize counts for true positives, false positives, and false negatives
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # Match detected circles with ground truth circles based on IoU threshold
        for gt_circle in ground_truth_circles:
            gt_center, gt_radius = gt_circle
            gt_area = np.pi * gt_radius ** 2
            matched = False
            for detected_circle in detected_circles:
                det_center, det_radius = detected_circle
                det_area = np.pi * det_radius ** 2
                iou_score = iou(gt_center, gt_radius, det_center, det_radius)
                if iou_score > 0.5:  # IoU threshold
                    true_positives += 1
                    matched = True
                    break
            if not matched:
                false_negatives += 1

        false_positives = len(detected_circles) - true_positives

        # Calculate precision, recall, and F1-score
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1_score": f1_score}

    except Exception as e:
        raise ValueError(f"Error evaluating model: {e}")

def iou(center1: Tuple[int, int], radius1: int, center2: Tuple[int, int], radius2: int) -> float:
    """
    Calculate the Intersection over Union (IoU) of two circles.

    Parameters:
        center1 (Tuple[int, int]): Center coordinates of the first circle.
        radius1 (int): Radius of the first circle.
        center2 (Tuple[int, int]): Center coordinates of the second circle.
        radius2 (int): Radius of the second circle.

    Returns:
        float: Intersection over Union (IoU) score.
    """
    x1, y1 = center1
    x2, y2 = center2
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if distance >= radius1 + radius2:
        return 0.0  # Circles do not intersect
    elif distance <= abs(radius1 - radius2):
        return 1.0  # One circle completely contains the other
    else:
        angle1 = np.arccos((radius1 ** 2 + distance ** 2 - radius2 ** 2) / (2 * radius1 * distance))
        angle2 = np.arccos((radius2 ** 2 + distance ** 2 - radius1 ** 2) / (2 * radius2 * distance))
        intersection_area = radius1 ** 2 * angle1 + radius2 ** 2 * angle2 - 0.5 * radius1 * distance * np.sin(angle1 * 2) - 0.5 * radius2 * distance * np.sin(angle2 * 2)
        union_area = np.pi * (radius1 ** 2 + radius2 ** 2) - intersection_area
        return intersection_area / union_area

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

