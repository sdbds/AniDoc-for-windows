import os
import cv2
import numpy as np
import shutil

def process_videos_and_images(input_folder, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)


    files = os.listdir(input_folder)

    for file_name in files:
        file_path = os.path.join(input_folder, file_name)

   
        if file_name.lower().endswith('.mp4'):
    
            video = cv2.VideoCapture(file_path)
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

        
            output_video_path = os.path.join(output_folder, file_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            for frame in sampled_frames:
                out_video.write(frame)

            out_video.release()

          
            frame_folder = os.path.join(output_folder, os.path.splitext(file_name)[0])
            os.makedirs(frame_folder, exist_ok=True)

            for idx, frame in enumerate(sampled_frames):
                frame_image_path = os.path.join(frame_folder, f'frame_{idx:03d}.png')
                cv2.imwrite(frame_image_path, frame)

    
        elif file_name.lower().endswith('.png'):
            shutil.copy(file_path, os.path.join(output_folder, file_name))

if __name__ == '__main__':
    input_folder = 'input_video_folder'  
    output_folder = 'preprocessed_video_folders' 
    process_videos_and_images(input_folder, output_folder)
