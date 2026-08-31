from tabnanny import check
import os, json, time
from tqdm import tqdm

import genais

API_KEY_JSON_PATH = "APIKEY/api_key.json"
API_KEY_JSON = json.load(open(API_KEY_JSON_PATH, "r"))
GEMINI_API_KEY = API_KEY_JSON["Gemini_yong"]

data_dir = "data/reconstructed_videos"
data_dir = "data/clipped"
video_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".mp4")]

# model_name = "gemini-3.1-pro-preview"
model_name = "gemini-3.5-flash-lite"
agent = genais.AgentGemini(model_name=model_name, api_key=GEMINI_API_KEY)
output_json_path = f"output/inference_results_{model_name}_videos.json"

ignore_basefilenames = [
    "clipped_1_14_Semiautomatic_nail_gun_accidents",
    "clipped_0_13_overhead_drilling_v1",
    "clipped_0_15_Drill work for roof ceiling shorts viral youtubeshortvideo roof",
    "clipped_0_22_fall ceiling drilling videoshot",
    "clipped_165_203_How to FRAME a Wall - 3 EASY STEPS_1080p",
    "clipped_0_19_Hand taping inside corners is always fun! LEVEL5 knifes always make the job easier",
    "clipped_0_25_Finishing Drywall Butt Joints with LEVEL5 Hand Tools",
    "clipped_0_11_Hard working Malaysia constructionworker",
    "clipped_0_12_Amazing fastest work rebar tying skill",
    "clipped_0_15_GUARANTEE",
    "clipped_11_25_Rebar tying (2)"
]

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
    base_filename = os.path.basename(video_path).replace(".mp4", "")
    if base_filename in ignore_basefilenames:
        print(f"Skipping {video_path} as it is in the ignore list.")
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