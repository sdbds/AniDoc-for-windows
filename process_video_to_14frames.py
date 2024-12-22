import os
import cv2
import numpy as np
import shutil
import argparse
import sys

def process_video(input_path: str, output_folder: str) -> str:
    """Process a single video file to 14 frames"""
    os.makedirs(output_folder, exist_ok=True)

    video = cv2.VideoCapture(input_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_indices = np.linspace(0, frame_count - 1, num=14, dtype=int)
    sampled_frames = []
    current_frame = 0
    frames_read = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        if current_frame in frame_indices:
            sampled_frames.append(frame)
            frames_read += 1
            if frames_read == len(frame_indices):
                break
        current_frame += 1

    video.release()

    output_video_path = os.path.join(output_folder, os.path.basename(input_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in sampled_frames:
        out_video.write(frame)

    out_video.release()

    # Save individual frames
    frame_folder = os.path.join(output_folder, os.path.splitext(os.path.basename(input_path))[0])
    os.makedirs(frame_folder, exist_ok=True)

    for idx, frame in enumerate(sampled_frames):
        frame_image_path = os.path.join(frame_folder, f'frame_{idx:03d}.png')
        cv2.imwrite(frame_image_path, frame)

    return output_video_path

def main():
    parser = argparse.ArgumentParser(description='Process video to 14 frames')
    parser.add_argument('--input', type=str, required=True, help='Input video file')
    parser.add_argument('--output', type=str, required=True, help='Output folder path')
    
    args = parser.parse_args()
    
    try:
        output_path = process_video(args.input, args.output)
        print(f"Successfully processed video to: {output_path}")
    except Exception as e:
        print(f"Error processing video: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
