import sys
import os
import torch
import argparse
import cv2  # Import OpenCV for saving images
from PIL import Image
from datetime import datetime
import json

sys.path.append('..')
from submodules.mvdream_diffusers.pipeline_mvdream import MVDreamPipeline

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate multi-view images and save them.")
parser.add_argument("--model", type=str, default="ashawkey/mvdream-sd2.1-diffusers", help="Model name or path")  # Default model
parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for generation")
parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps")
parser.add_argument("--num_frames", type=int, default=4, help="Number of frames to generate")
args = parser.parse_args()

# Initialize the pipeline
pipe = MVDreamPipeline.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
pipe = pipe.to("cuda")

# Generate frames using the specified arguments
images = pipe(
    args.prompt,
    guidance_scale=args.guidance_scale,
    num_inference_steps=args.num_inference_steps,
    num_frames=args.num_frames,
)

# Create a folder to save the images
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"mv_out/{args.prompt.replace(' ', '_')}_{timestamp}"
os.makedirs(output_folder, exist_ok=True)

# Save each image to the folder
image_paths = []
frame_size = None  # To store the size of the frames for the video
for i, img in enumerate(images):
    img_path = os.path.join(output_folder, f"frame_{i + 1}.png")
    # Scale the image from [0, 1] to [0, 255] and convert to uint8
    img_uint8 = (img * 255).astype("uint8")
    # Convert the image from RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, img_bgr)  # Save the image using OpenCV
    image_paths.append(img_path)
    if frame_size is None:
        frame_size = (img_bgr.shape[1], img_bgr.shape[0])  # (width, height)

# Save configuration to a JSON file
config = {
    "model": args.model,
    "prompt": args.prompt,
    "guidance_scale": args.guidance_scale,
    "num_inference_steps": args.num_inference_steps,
    "num_frames": args.num_frames,
    "output_folder": output_folder,
    "image_paths": image_paths,
}
config_path = os.path.join(output_folder, "config.json")
with open(config_path, "w") as config_file:
    json.dump(config, config_file, indent=4)

# Create a grid (1 x num_frames) of the images
grid_width = sum(img.shape[1] for img in images)  # Use shape[1] for width
grid_height = max(img.shape[0] for img in images)  # Use shape[0] for height
grid = Image.new("RGB", (grid_width, grid_height))

x_offset = 0
for img in images:
    # Scale the image from [0, 1] to [0, 255] and convert to uint8
    img_uint8 = (img * 255).astype("uint8")
    # Convert NumPy array to PIL Image
    img_pil = Image.fromarray(img_uint8)
    grid.paste(img_pil, (x_offset, 0))
    x_offset += img.shape[1]

# Save the grid to the folder
grid_path = os.path.join(output_folder, "grid.png")
grid.save(grid_path, format="PNG")

# Create a video from the frames
video_path = os.path.join(output_folder, "output_video.mp4")
fps = 5  # Frames per second
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

for img_path in image_paths:
    frame = cv2.imread(img_path)  # Read each frame
    video_writer.write(frame)  # Write the frame to the video

video_writer.release()  # Finalize the video

print(f"Images saved to folder: {output_folder}")
print(f"Grid saved to: {grid_path}")
print(f"Configuration saved to: {config_path}")
print(f"Video saved to: {video_path}")