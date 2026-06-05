from tabnanny import check
import os, json, time
from tqdm import tqdm

import genais

API_KEY_JSON_PATH = "APIKEY/api_key.json"
API_KEY_JSON = json.load(open(API_KEY_JSON_PATH, "r"))
GEMINI_API_KEY = API_KEY_JSON["Gemini_yong"]

data_dir = "data/reconstructed_videos"
video_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".mp4")]

# model_name = "gemini-3.1-pro-preview"
model_name = "gemini-3.1-flash-lite-preview"
agent = genais.AgentGemini(model_name=model_name, api_key=GEMINI_API_KEY)
output_json_path = f"output/inference_results_{model_name}_videos.json"

processed_video_paths = set()
if os.path.exists(output_json_path):
    with open(output_json_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                processed_video_paths.add(result["video_path"])

for video_path in tqdm(video_paths, desc="Processing videos"):
    if video_path in processed_video_paths:
        continue
    
    video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    # print(f"Processing {video_path} (size: {video_size_mb:.2f} MB)")
    segments, inference_cost, inference_time = agent.inference_one_video(video_path)
    segments = [(seg.action, seg.start_second) for seg in segments]
    result = {
        "video_path": video_path,
        "segments": segments,
        "inference_cost": inference_cost,
        "inference_time": inference_time
    }
    with open(output_json_path, "a") as f:
        json.dump(result, f)
        f.write("\n")
    time.sleep(3)