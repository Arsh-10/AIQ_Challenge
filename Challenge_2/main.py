import pandas as pd
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import sqlite3
import base64
from typing import List, Dict, Union

app = FastAPI()

DATABASE = "Your_database.db"

# Initialize the database by creating the images table if it doesn't exist
def init_db():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS images (depth REAL, pixels TEXT)''')
        conn.commit()
        conn.close()
    except Exception as e:
        return {"error": str(e)}

# Root endpoint to initialize the database
@app.get("/")
def read_root():
    init_db()

# API endpoint to upload a CSV file, resize the images, and store them in the database
@app.post("/process_images")
async def process_images(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        resized_images = resize_and_store_images(df)
        return {"info": "CSV uploaded and images processed successfully", "resized_images": resized_images}
    except Exception as e:
        return {"error": str(e)}

# API endpoint to retrieve frames based on depth range and apply a color map to each frame
@app.get("/frames")
def get_frames(depth_min: int = Query(...), depth_max: int = Query(...)):
    try:
        frames = retrieve_frames(depth_min, depth_max)
        processed_frames = []
        for frame in frames:
            image_data = base64.b64decode(frame["pixels"])
            image = np.frombuffer(image_data, dtype=np.uint8).reshape((1, 150))
            processed_frame = {"depth": frame["depth"], "pixels": apply_color_map(image).tolist()}
            processed_frames.append(processed_frame)
        return JSONResponse(content={"frames": processed_frames})
    except Exception as e:
        return {"error": str(e)}

def resize_and_store_images(df: pd.DataFrame) -> None:
    """
    Resizes images and stores them in the database.

    Args:
        df (pd.DataFrame): The DataFrame containing image data.

    Returns:
        None
    """
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        for index, row in df.iterrows():
            depth = row[0]
            if pd.isna(depth):  # Check if depth is NaN
                continue  # Skip this row if depth is NaN
            depth = float(depth)  # Convert depth to float
            pixels = row[1:].values.astype(np.uint8)  # Convert to uint8
            resized_image = cv2.resize(pixels.reshape(1, -1), (150, 1))
            encoded_image = base64.b64encode(resized_image).decode('utf-8')
            cursor.execute("INSERT INTO images (depth, pixels) VALUES (?, ?)", (depth, encoded_image))
        conn.commit()
        conn.close()
    except Exception as e:
        return {"error": str(e)}

# Function to retrieve frames from the database based on depth range
def retrieve_frames(depth_min: int, depth_max: int) -> List[Dict[str, Union[int, str]]]:
    """
    Retrieves frames from the database where the depth is within the specified range (depth_min to depth_max).

    Args:
        depth_min (int): The minimum depth value for filtering frames.
        depth_max (int): The maximum depth value for filtering frames.

    Returns:
        List[Dict[str, Union[int, str]]]: List of dictionaries containing depth and pixel data for each frame.
    """
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM images WHERE depth BETWEEN ? AND ?", (depth_min, depth_max))
        rows = cursor.fetchall()
        conn.close()
        frames = [{"depth": row[0], "pixels": row[1]} for row in rows]
        return frames
    except Exception as e:
        return {"error": str(e)}

# Function to apply a color map to a given image
def apply_color_map(image: np.ndarray) -> np.ndarray:
    """
    Applies a color map (using OpenCV's COLORMAP_JET) to the given image to enhance visualization.

    Args:
        image (np.ndarray): The input image to which the color map will be applied.

    Returns:
        np.ndarray: The color-mapped image.
    """
    return cv2.applyColorMap(image, cv2.COLORMAP_JET)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
