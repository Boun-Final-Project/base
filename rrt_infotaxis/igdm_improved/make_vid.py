import cv2
import os
import re

def natural_sort_key(s):
    """
    Helper function to sort file names numerically (e.g., step_2.png comes before step_10.png).
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def create_video_from_images(image_folder, output_video_file, fps=1):
    """
    Converts a folder of images into a video file.
    
    Parameters:
    - image_folder: Path to the directory containing images.
    - output_video_file: Name of the output video file (e.g., 'result.mp4').
    - fps: Frames per second. fps=1 means each image shows for 1 second.
    """
    
    # 1. Get all image files
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    
    # 2. Sort images numerically to ensure correct step order
    images.sort(key=natural_sort_key)
    
    if not images:
        print(f"Skipping {image_folder}: No images found.")
        return

    # 3. Read the first image to determine video frame size
    first_frame_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_frame_path)
    height, width, layers = frame.shape
    size = (width, height)

    # 4. Initialize VideoWriter
    # 'mp4v' is the codec for .mp4 files. 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_file, fourcc, fps, size)

    print(f"Processing {len(images)} images in '{image_folder}'...")

    # 5. Write images to video
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        img = cv2.imread(image_path)
        
        # Ensure image was read correctly
        if img is not None:
            # Resize if necessary to match the first frame (optional, but good safety)
            if (img.shape[1], img.shape[0]) != size:
                img = cv2.resize(img, size)
            out.write(img)
    
    out.release()
    print(f"Done! Saved video to: {output_video_file}\n")

if __name__ == "__main__":
    # ================= CONFIGURATION =================
    
    # CHANGE THIS to the path holding your 6 directories
    # Example: "/home/user/project/simulation_outputs"
    base_dir = "." 
    
    # Optional: List specific folder names if you don't want to scan everything
    # folders_to_process = ["folder1", "folder2", "folder3"]
    # If None, it will process all subdirectories in base_dir
    folders_to_process = None 
    
    # Duration: 1 frame per second = 1 second per image
    FRAME_RATE = 1 
    
    # =================================================

    # If folders_to_process is None, get all directories in base_dir
    if folders_to_process is None:
        folders_to_process = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for folder in folders_to_process:
        full_folder_path = os.path.join(base_dir, folder)
        
        # Output video name based on folder name
        output_name = f"{folder}_timelapse.mp4"
        
        # Run the conversion
        create_video_from_images(full_folder_path, output_name, fps=FRAME_RATE)