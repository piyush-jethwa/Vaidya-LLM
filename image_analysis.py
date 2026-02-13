import cv2
import numpy as np
from PIL import Image

def analyze_image_colors(image_path, max_size=1000, quality=0.8):
    """Analyze dominant colors in an image with memory optimization"""
    try:
        # First check image validity with grayscale read
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("Could not read image file")
            
        # Calculate resize scale if needed
        height, width = gray.shape
        scale = 1.0
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            
        # Read full color image at appropriate quality
        img = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_2 if scale < 0.5 else cv2.IMREAD_COLOR)
        if scale < 1.0:
            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Sample pixels for faster processing
        pixels = img.reshape((-1, 3))
        if len(pixels) > 10000:  # Sample if too many pixels
            pixels = pixels[np.random.choice(len(pixels), 10000, replace=False)]
        
        # Get dominant colors using k-means with reduced iterations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
        _, _, centers = cv2.kmeans(
            pixels.astype(np.float32),
            3, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Convert centers to hex colors
        colors = [f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}" for c in centers]
        return {
            "dominant_colors": colors,
            "original_size": (width, height),
            "processed_size": img.shape[:2]
        }
        
    except Exception as e:
        raise ValueError(f"Image analysis error: {str(e)}")

def detect_edges(image_path):
    """Detect edges in an image"""
    try:
        img = cv2.imread(image_path, 0)
        if img is None:
            raise ValueError("Could not read image file")
            
        edges = cv2.Canny(img, 100, 200)
        return edges
        
    except Exception as e:
        raise ValueError(f"Edge detection error: {str(e)}")
