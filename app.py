import os
import shutil
import uuid
import subprocess
import gradio as gr
import cv2
import sys
from glob import glob
from pathlib import Path

# 获取当前Python解释器路径
PYTHON_EXECUTABLE = sys.executable

def normalize_path(path: str) -> str:
    """标准化路径，将Windows路径转换为正斜杠形式"""
    return str(Path(path).resolve()).replace('\\', '/')

def check_video_frames(video_path: str) -> int:
    """检查视频帧数"""
    video_path = normalize_path(video_path)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def preprocess_video(video_path: str) -> str:
    """预处理视频到14帧"""
    try:
        video_path = normalize_path(video_path)
        unique_id = str(uuid.uuid4())
        temp_dir = "outputs"
        output_dir = os.path.join(temp_dir, f"processed_{unique_id}")
        output_dir = normalize_path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing video: {video_path}")
        print(f"Output directory: {output_dir}")
        
        # 调用process_video_to_14frames.py处理视频
        result = subprocess.run(
            [
                PYTHON_EXECUTABLE, "process_video_to_14frames.py",
                "--input", video_path,
                "--output", output_dir
            ],
            check=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            print(f"Preprocessing stdout: {result.stdout}")
        if result.stderr:
            print(f"Preprocessing stderr: {result.stderr}")
        
        # 获取处理后的视频路径
        processed_videos = glob(os.path.join(output_dir, "*.mp4"))
        if not processed_videos:
            raise gr.Error("Failed to process video: No output video found")
        return normalize_path(processed_videos[0])
    except subprocess.CalledProcessError as e:
        print(f"Preprocessing stderr: {e.stderr}")
        raise gr.Error(f"Failed to preprocess video: {e.stderr}")
    except Exception as e:
        raise gr.Error(f"Error during video preprocessing: {str(e)}")

def generate(control_sequence, ref_image):
    try:
        # 验证输入文件是否存在
        control_sequence = normalize_path(control_sequence)
        ref_image = normalize_path(ref_image)
        
        if not os.path.exists(control_sequence):
            raise gr.Error(f"Control sequence file not found: {control_sequence}")
        if not os.path.exists(ref_image):
            raise gr.Error(f"Reference image file not found: {ref_image}")
            
        # 创建输出目录
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        unique_id = str(uuid.uuid4())
        result_dir = os.path.join(output_dir, f"results_{unique_id}")
        result_dir = normalize_path(result_dir)
        os.makedirs(result_dir, exist_ok=True)
        
        print(f"Input control sequence: {control_sequence}")
        print(f"Input reference image: {ref_image}")
        print(f"Output directory: {result_dir}")
        
        # 检查视频帧数
        frame_count = check_video_frames(control_sequence)
        if frame_count != 14:
            print(f"Video has {frame_count} frames, preprocessing to 14 frames...")
            control_sequence = preprocess_video(control_sequence)
            print(f"Preprocessed video saved to: {control_sequence}")
        
        # 运行推理命令
        print(f"Running inference...")
        result = subprocess.run(
            [
                PYTHON_EXECUTABLE, "scripts_infer/anidoc_inference.py",
                "--all_sketch",
                "--matching",
                "--tracking",
                "--control_image", control_sequence,
                "--ref_image", ref_image,
                "--output_dir", result_dir,
                "--max_point", "10",
            ],
            check=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            print(f"Inference stdout: {result.stdout}")
        if result.stderr:
            print(f"Inference stderr: {result.stderr}")

        # 搜索输出视频
        output_video = glob(os.path.join(result_dir, "*.mp4"))
        print(f"Found output videos: {output_video}")
        
        if output_video:
            output_video_path = normalize_path(output_video[0])
            print(f"Returning output video: {output_video_path}")
        else:
            raise gr.Error("No output video generated")
            
        # 清理临时文件
        temp_dirs = glob("outputs/processed_*")
        for temp_dir in temp_dirs:
            if os.path.isdir(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    print(f"Warning: Failed to clean up temp directory {temp_dir}: {str(e)}")
        
        return output_video_path
        
    except subprocess.CalledProcessError as e:
        print(f"Inference stderr: {e.stderr}")
        raise gr.Error(f"Error during inference: {e.stderr}")
    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")

css="""
div#col-container{
    margin: 0 auto;
    max-width: 982px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# AniDoc: Animation Creation Made Easier")
        gr.Markdown("AniDoc colorizes a sequence of sketches based on a character design reference with high fidelity, even when the sketches significantly differ in pose and scale.")
        gr.HTML("""
        <div style="display:flex;column-gap:4px;">
            <a href="https://github.com/yihao-meng/AniDoc">
                <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
            </a> 
            <a href="https://yihao-meng.github.io/AniDoc_demo/">
                <img src='https://img.shields.io/badge/Project-Page-green'>
            </a>
            <a href="https://arxiv.org/pdf/2412.14173">
                <img src='https://img.shields.io/badge/ArXiv-Paper-red'>
            </a>
        </div>
        """)
        with gr.Row():
            with gr.Column():
                control_sequence = gr.Video(label="Control Sequence", format="mp4")
                ref_image = gr.Image(label="Reference Image", type="filepath")
                submit_btn = gr.Button("Submit")
            with gr.Column():
                video_result = gr.Video(label="Result")

                gr.Examples(
                    examples = [
                        ["data_test/sample5.mp4", "data_test/sample5.png"],
                        ["data_test/sample1.mp4", "data_test/sample1.png"],
                        ["data_test/sample2.mp4", "data_test/sample2.png"],
                        ["data_test/sample3.mp4", "data_test/sample3.png"],
                        ["data_test/sample4.mp4", "data_test/sample4.png"]
                    ],
                    inputs = [control_sequence, ref_image]
        )
        
        submit_btn.click(
        fn = generate,
        inputs = [control_sequence, ref_image],
        outputs = [video_result]
        )

demo.queue().launch(inbrowser=True,show_api=False, show_error=True)