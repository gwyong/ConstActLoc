import os, json, time
from tqdm import tqdm

import genais

API_KEY_JSON_PATH = "APIKEY/api_key.json"
API_KEY_JSON = json.load(open(API_KEY_JSON_PATH, "r"))
OPENAI_API_KEY = API_KEY_JSON["OpenAI_UM"] # OpenAI_yong
CLAUDE_API_KEY = API_KEY_JSON["Anthropic_yong"]
GEMINI_API_KEY = API_KEY_JSON["Gemini_yong"]

fps = 1
data_dir = f"data/frames_fps{fps}"
frame_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jpg")]

#model_name = "gpt-5.6-terra" # "gpt-5.4"
# model_name = "gpt-5.6-luna" # "gpt-5.4-mini"
# agent = genais.AgentOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)

model_name = "claude-sonnet-5" # "claude-sonnet-4-6"
# model_name = "claude-haiku-4-5-20251001"
agent = genais.AgentClaude(model_name=model_name, api_key=CLAUDE_API_KEY)

# model_name = "gemini-3.1-pro-preview"
# agent = genais.AgentGemini(model_name=model_name, api_key=GEMINI_API_KEY)

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
output_json_path = f"output/inference_results_{model_name}.json"
processed_frame_paths = set()
if os.path.exists(output_json_path):
    with open(output_json_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                processed_frame_paths.add(result["frame_path"])

for frame_path in tqdm(frame_paths, desc="Processing frames"):
    if frame_path in processed_frame_paths:
        continue
    base_filename = os.path.basename(frame_path).replace(".jpg", "")
    if any(base_filename.startswith(ignore_name) for ignore_name in ignore_basefilenames):
        continue

    action, inference_cost, inference_time = agent.inference_one_frame(frame_path)
    result = {
        "frame_path": frame_path,
        "action": action,
        "inference_cost": inference_cost,
        "inference_time": inference_time
    }
    with open(output_json_path, "a") as f:
        json.dump(result, f)
        f.write("\n")
    time.sleep(0.1)