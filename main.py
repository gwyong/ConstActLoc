import os, json, time
from tqdm import tqdm

import genais

API_KEY_JSON_PATH = "APIKEY/api_key.json"
API_KEY_JSON = json.load(open(API_KEY_JSON_PATH, "r"))
OPENAI_API_KEY = API_KEY_JSON["OpenAI_yong"]
CLAUDE_API_KEY = API_KEY_JSON["Anthropic_yong"]

fps = 1
data_dir = f"data/frames_fps{fps}"
frame_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jpg")]

# model_name = "gpt-5.4"
# agent = genais.AgentOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)

model_name = "claude-sonnet-4-6"
agent = genais.AgentClaude(model_name=model_name, api_key=CLAUDE_API_KEY)

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