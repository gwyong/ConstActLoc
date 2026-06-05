import os, json
import pandas as pd
from tqdm import tqdm

import genais
API_KEY_JSON_PATH = "APIKEY/api_key.json"
API_KEY_JSON = json.load(open(API_KEY_JSON_PATH, "r"))
OPENAI_API_KEY = API_KEY_JSON["OpenAI_yong"]
CLAUDE_API_KEY = API_KEY_JSON["Anthropic_yong"]
GEMINI_API_KEY = API_KEY_JSON["Gemini_yong"]

gt_classes = pd.read_csv("data/action_classes.csv")["class"].tolist()
df_pred = pd.read_json("output/inference_results_gpt-5.4.json", lines=True)
# df_pred = pd.read_json("output/inference_results_claude-sonnet-4-6.json", lines=True)
unique_classes = (
    df_pred["action"]
    .dropna()
    .astype(str)
    .str.strip()
    .loc[lambda s: s != ""]
    .sort_values()
    .unique()
)

# check weird classes in unique_classes but not in gt_classes
weird_classes = [c for c in unique_classes if c not in gt_classes]
print("Weird classes:", weird_classes)

# check gt_classes that are not in unique_classes
missing_classes = [c for c in gt_classes if c not in unique_classes]
print("Missing classes:", missing_classes)

source_class = "lift/carry window/sheets"
# check how many rows in df_pred have action == source_class
num_source_class = (df_pred["action"] == source_class).sum()
print(f"Number of rows with action == '{source_class}': {num_source_class}")

model_name = "gpt-5.4"
agent = genais.AgentOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)

# model_name = "claude-sonnet-4-6"
# agent = genais.AgentClaude(model_name=model_name, api_key=CLAUDE_API_KEY)

for idx, row in tqdm(df_pred.iterrows(), total=len(df_pred), desc="Correcting classes"):
    action = row["action"]
    if pd.isna(action) or action.strip() == "":
        continue
    if action == source_class:
        frame_path = row["frame_path"]
        # \\ to /
        frame_path = frame_path.replace("\\", "/")
        action, inference_cost, inference_time = agent.inference_one_frame(frame_path)
        result = {
            "frame_path": frame_path,
            "action": action,
            "inference_cost": inference_cost,
            "inference_time": inference_time
        }
        # replace the row in df_pred with the new result
        df_pred.at[idx, "action"] = action
        df_pred.at[idx, "inference_cost"] = inference_cost
        df_pred.at[idx, "inference_time"] = inference_time

# save the corrected df_pred to a new json file
output_json_path = f"output/inference_results_{model_name}.json"
df_pred.to_json(output_json_path, orient="records", lines=True)

# check weird classes again after correction
unique_classes_after = (
    df_pred["action"]
    .dropna()
    .astype(str)
    .str.strip()
    .loc[lambda s: s != ""]
    .sort_values()
    .unique()
)
weird_classes_after = [c for c in unique_classes_after if c not in gt_classes]
print("Weird classes after correction:", weird_classes_after)
missing_classes_after = [c for c in gt_classes if c not in unique_classes_after]
print("Missing classes after correction:", missing_classes_after)