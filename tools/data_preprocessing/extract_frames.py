"""
Extract frames from a video.

Author: Stéphane KPOVIESSI
At Sylvagreg company
Big data engineer student
"""

import cv2
import os
import argparse

def extract_frames(video_path, output_folder, frame_rate):
    """
    Extracts frames from a video at a specified frame rate.

    Parameters:
    video_path (str): Path to the input video file.
    output_folder (str): Folder to save the extracted frames.
    frame_rate (int): Frame rate to extract frames (frames per second).
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video FPS: {fps}")

    # Calculate the interval between frames to extract
    interval = int(fps / frame_rate)

    frame_idx = 0
    extracted_frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            frame_name = os.path.join(output_folder, f"frame_{extracted_frame_idx:04d}.png")
            cv2.imwrite(frame_name, frame)
            extracted_frame_idx += 1

        frame_idx += 1

    cap.release()
    print(f"Extracted {extracted_frame_idx} frames.")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_folder", type=str, help="Folder to save the extracted frames.")
    parser.add_argument("--frame_rate", type=int, default=5, help="Frame rate to extract frames (frames per second).")
    
    args = parser.parse_args()
    
    extract_frames(args.video_path, args.output_folder, args.frame_rate)

if __name__ == "__main__":
    main()
