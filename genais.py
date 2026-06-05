import os, glob, time, json, re, random, shutil, pickle
import base64, cv2
import pandas as pd

from typing import Dict, Any, List, Literal
from pydantic import BaseModel, conlist, Field
from copy import deepcopy
from datetime import datetime

from openai import OpenAI
import anthropic

from google import genai
from google.genai import types

import utils

API_KEY_JSON_PATH = "APIKEY/api_key.json"
API_KEY_JSON = json.load(open(API_KEY_JSON_PATH, "r"))
OPENAI_API_KEY = API_KEY_JSON["OpenAI_yong"]
CLAUDE_API_KEY = API_KEY_JSON["Anthropic_yong"]
GEMINI_API_KEY = API_KEY_JSON["Gemini_yong"]
df_class_path = "data/action_classes.csv"
df_class = pd.read_csv(df_class_path)
action_classes = df_class["class"].tolist()

# with open("data/action_classes.pkl", "rb") as f:
#     action_classes = pickle.load(f)
if "none" not in action_classes:
    action_classes.append("none")

BASIC_PROMPT = f"""
# ROLE:
You are an expert construction manager analyzing video frames from construction site videos.

# INSTRUCTION:
Determine the worker's current action from the provided frame or frames.

An action should be identified ONLY when the worker is actively performing a construction task through direct physical interaction with relevant work entities, such as:
- tools
- materials
- construction equipment
- building components

Direct physical interaction means the worker's hand(s) are visibly in contact with the object(s) involved in the task.

Return "none" in the following cases:
- the worker is preparing to start a task
- the worker is observing, waiting, or idle
- the worker's hands are not in active contact with relevant work objects
- the visible action does not clearly match any provided action options
- the worker is performing a non-target action outside the provided action list

Choose ONLY from the provided action options below:
{action_classes}

Be conservative. If the action is ambiguous or insufficiently visible, return "none".

# OUTPUT FORMAT:
Return a valid JSON object only.
Key: "action"
Value: One action from the provided choices or "none"

# EXAMPLE:
Input: A frame showing a worker using a hammer to drive a nail into a piece of wood
Output: "action": "hammer"
"""

api_pricing_dict = {
    "gpt-5.5": {"input_cost_per_1Mtks": 5.0, "cache_cost_per_1Mtks": 0.5, "output_cost_per_1Mtks": 30.0}, # 2025.12.01 Knowledge Cutoff
    "gpt-5.4": {"input_cost_per_1Mtks": 2.5, "cache_cost_per_1Mtks": 0.175, "output_cost_per_1Mtks": 15.0}, # 2025.08.31
    "gpt-5.4-mini": {"input_cost_per_1Mtks": 0.75, "cache_cost_per_1Mtks": 0.08, "output_cost_per_1Mtks": 4.5}, # 2025.08.31
    # "gpt-5.2": {"input_cost_per_1Mtks": 1.75, "cache_cost_per_1Mtks": 0.175, "output_cost_per_1Mtks": 14.0},
    # "gpt-5.1": {"input_cost_per_1Mtks": 1.25, "cache_cost_per_1Mtks": 0.50, "output_cost_per_1Mtks": 10.0},
    # "gpt-4.1": {"input_cost_per_1Mtks": 2.0, "cache_cost_per_1Mtks": 0.50, "output_cost_per_1Mtks": 8.0},
    "gpt-4o": {"input_cost_per_1Mtks": 2.5, "cache_cost_per_1Mtks": 1.25, "output_cost_per_1Mtks": 10.0}, # 2023.10.01
    "gpt-4o-mini": {"input_cost_per_1Mtks": 0.15, "cache_cost_per_1Mtks": 0.08, "output_cost_per_1Mtks": 0.6}, # 2023.10.01
    # "o1": {"input_cost_per_1Mtks": 15.0, "cache_cost_per_1Mtks": 7.50, "output_cost_per_1Mtks": 60.0},
    # "claude-3-5-sonnet-latest": {"input_cost_per_1Mtks": 3.0, "cache_cost_per_1Mtks": 3.75, "output_cost_per_1Mtks": 15.0},
    # "claude-3-7-sonnet-latest": {"input_cost_per_1Mtks": 3.0, "cache_cost_per_1Mtks": 3.75, "output_cost_per_1Mtks": 15.0}, Retired!
    # "claude-sonnet-4-20250514": {"input_cost_per_1Mtks": 3.0, "cache_cost_per_1Mtks": 3.75, "output_cost_per_1Mtks": 15.0}, 
    "claude-sonnet-4-5-20250929": {"input_cost_per_1Mtks": 3.0, "cache_cost_per_1Mtks": 3.75, "output_cost_per_1Mtks": 15.0}, # Will be retired 2026.09.29
    # "claude-4-5-sonnet-latest": {"input_cost_per_1Mtks": 3.0, "cache_cost_per_1Mtks": 3.75, "output_cost_per_1Mtks": 15.0},
    "claude-sonnet-4-6": {"input_cost_per_1Mtks": 3.0, "cache_cost_per_1Mtks": 3.75, "output_cost_per_1Mtks": 15.0}, # 2025.08
    "claude-haiku-4-5-20251001": {"input_cost_per_1Mtks": 1.0, "cache_cost_per_1Mtks": 3.75, "output_cost_per_1Mtks": 5.0}, # 2025.02
    "claude-opus-4-7": {"input_cost_per_1Mtks": 5.0, "cache_cost_per_1Mtks": 3.75, "output_cost_per_1Mtks": 25.0}, # 2026.01
    "gemini-3.1-pro-preview": {"input_cost_per_1Mtks": 2.0, "cache_cost_per_1Mtks": 0.2, "output_cost_per_1Mtks": 12.0}, # 2025.01
    "gemini-3.1-flash-lite-preview": {"input_cost_per_1Mtks": 0.25, "cache_cost_per_1Mtks": 0.025, "output_cost_per_1Mtks": 1.5}, # 2025.01
    "gemini-3-pro-preview": {"input_cost_per_1Mtks": 2.0, "cache_cost_per_1Mtks": 0.2, "output_cost_per_1Mtks": 12.0},
    "gemini-3-flash-preview": {"input_cost_per_1Mtks": 0.25, "cache_cost_per_1Mtks": 0.025, "output_cost_per_1Mtks": 1.5}, # 2025.01
    # "gemini-2.5-pro-preview-06-05": {"input_cost_per_1Mtks": 1.25, "cache_cost_per_1Mtks": 0.00, "output_cost_per_1Mtks": 10.0},
    "gemini-2.5.pro": {"input_cost_per_1Mtks": 1.25, "cache_cost_per_1Mtks": 0.125, "output_cost_per_1Mtks": 10.0},
    "gemini-2.5-flash-lite": {"input_cost_per_1Mtks": 0.1, "cache_cost_per_1Mtks": 0.01, "output_cost_per_1Mtks": 0.4},
    # "gemini-2.0-flash-exp": {"input_cost_per_1Mtks": 1.25, "cache_cost_per_1Mtks": 0.00, "output_cost_per_1Mtks": 5.0},
    # "gemini-2.0-flash": {"input_cost_per_1Mtks": 0.1, "cache_cost_per_1Mtks": 0.4, "output_cost_per_1Mtks": 1.0},
    # "gemini-1.5-pro": {"input_cost_per_1Mtks": 1.25, "cache_cost_per_1Mtks": 0.3125, "output_cost_per_1Mtks": 5.0},
}

top_k = 1
class StructuredResponse(BaseModel):
    action: conlist(Literal[tuple(action_classes)], min_length=top_k, max_length=top_k)

ActionLiteral = Literal[tuple(action_classes)]
class ActionSegment(BaseModel):
    action: ActionLiteral
    start_second: int = Field(ge=0)


class StructuredVideoResponse(BaseModel):
    segments: list[ActionSegment]

VIDEO_PROMPT = f"""
# ROLE:
You are an expert construction manager analyzing construction site videos.

# INSTRUCTION:
Analyze the entire video and identify when each worker action starts.

An action should be identified ONLY when the worker is actively performing a construction task through direct physical interaction with relevant work entities, such as:
- tools
- materials
- construction equipment
- building components

Direct physical interaction means the worker's hand(s) are visibly in contact with the object(s) involved in the task.

Return "none" when:
- the worker is preparing to start a task
- the worker is observing, waiting, walking, or idle
- the worker's hands are not in active contact with relevant work objects
- the visible action does not clearly match any provided action options
- the worker is performing a non-target action outside the provided action list

Choose ONLY from the provided action options below:
{action_classes}

You must detect action change points.
For each continuous action segment, return ONLY:
- the action label
- the integer second when that action starts

The end time of each action segment will be inferred as the next segment's start_second.
Therefore, do NOT return end times.

If an action is already happening at the beginning of the video, use start_second = 0.

Be conservative.
If the action is ambiguous or insufficiently visible, return "none".

# OUTPUT FORMAT:
Return a valid JSON object only.

Schema:
{{
  "segments": [
    ["action_name", start_second]
  ]
}}

Rules:
- "segments" must be a list of tuples/lists.
- Each item must contain exactly two values: [action, start_second].
- action must be one of the provided action options or "none".
- start_second must be an integer.
- start_second can be 0.
- start_second values must be sorted in ascending order.
- Do not include duplicate consecutive actions. If the same action continues, return only its first start_second.
- Do not include explanations, markdown, or extra text.

# EXAMPLE:
Input: A video where the worker is idle from 0s, starts hammering at 3s, then starts carrying material at 8s.
Output:
{{
  "segments": [
    ["none", 0],
    ["hammer", 3],
    ["carry_material", 8]
  ]
}}
"""

class AgentOpenAI():
    def __init__(self, logger=None, model_name="gpt-5.4", api_key=OPENAI_API_KEY, region=None):
        """
        Initialize OpenAI Agent class.
        Args:
            logger (Logger): Logger instance for logging information. NOTE: Unused.
            model_name (str): Name of the model to use. Defaults to "gpt-5.4".
            api_key (str): API key for authentication.
            region (str, optional): Region for the API. Defaults to None. NOTE: Unused.
            size limit: Up to 512 MB
        """
        self.logger = logger
        self.model_name = model_name
        self.pricing_dict = api_pricing_dict[model_name]
        self.api_key = api_key
        self.region = region
        self.agent = OpenAI(api_key=api_key)
        self.prompt = BASIC_PROMPT
    
    def inference_one_frame(self, frame_path, prompt=BASIC_PROMPT):
        base64_frame = utils.encode_image(frame_path)
        openai_input = [
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": prompt },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_frame}",
                            "detail": "auto"
                        },
                    ],
                }
            ]
        start_time = time.time()
        # response = self.agent.responses.create(model=self.model_name, input=openai_input)
        response = self.agent.responses.parse(model=self.model_name, input=openai_input, text_format=StructuredResponse)
        end_time = time.time()
        action = response.output_parsed.action[0]
        inference_cost = (response.usage.input_tokens*self.pricing_dict["input_cost_per_1Mtks"] + response.usage.output_tokens*self.pricing_dict["output_cost_per_1Mtks"])/1000000
        inference_time = end_time - start_time
        return action, inference_cost, inference_time
    

class AgentClaude():
    def __init__(self, logger=None, model_name="claude-sonnet-4-6", api_key=OPENAI_API_KEY, region=None):
        self.logger = logger
        self.model_name = model_name
        self.pricing_dict = api_pricing_dict[model_name]
        self.api_key = api_key
        self.region = region
        self.agent = anthropic.Anthropic(api_key=api_key)
        self.prompt = BASIC_PROMPT
        self.max_tokens = 1024
        self.image_media_type = "image/jpeg"

    def inference_one_frame(self, frame_path, prompt=BASIC_PROMPT):
        base64_frame = utils.encode_image(frame_path)
        claude_input = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": self.image_media_type,
                            "data": base64_frame,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
        ]
        start_time = time.time()
        # message = self.agent.messages.create(model=self.model_name, max_tokens=self.max_tokens, messages=claude_input)
        message = self.agent.messages.parse(model=self.model_name, max_tokens=self.max_tokens, messages=claude_input, output_format=StructuredResponse)
        end_time = time.time()
        action = message.parsed_output.action[0]
        inference_cost = (message.usage.input_tokens*self.pricing_dict["input_cost_per_1Mtks"] + message.usage.output_tokens*self.pricing_dict["output_cost_per_1Mtks"])/1000000
        inference_time = end_time - start_time
        return action, inference_cost, inference_time
    

class AgentGemini():
    def __init__(self, logger=None, model_name="gemini-3.1-pro-preview", api_key=GEMINI_API_KEY, region=None):
        self.logger = logger
        self.model_name = model_name
        self.pricing_dict = api_pricing_dict[model_name]
        self.api_key = api_key
        self.region = region
        self.agent = genai.Client(api_key=api_key)
        self.prompt = BASIC_PROMPT
        self.max_tokens = 1024
        self.image_media_type = "image/jpeg"

    def inference_one_frame(self, frame_path, prompt=BASIC_PROMPT):
        with open(frame_path, 'rb') as f:
            image_bytes = f.read()
        start_time = time.time()
        
        response = self.agent.models.generate_content(
        model=self.model_name,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=self.image_media_type,
            ),
            prompt,
        ],
        config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=StructuredResponse, # Pass the Pydantic class directly
        ),
        )
        end_time = time.time()
        
        parsed = StructuredResponse.model_validate_json(response.text)
        action = parsed.action[0]
        inference_cost = (response.usage_metadata.prompt_token_count*self.pricing_dict["input_cost_per_1Mtks"] + response.usage_metadata.candidates_token_count*self.pricing_dict["output_cost_per_1Mtks"])/1000000
        inference_time = end_time - start_time
        return action, inference_cost, inference_time
    
    def inference_one_video(self, video_path, prompt=BASIC_PROMPT):
        video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        # print(f"Processing {video_path} (size: {video_size_mb:.2f} MB)")
        
        start_time = time.time()
        if video_size_mb < 20:
            video_bytes = open(video_path, 'rb').read()
            response = self.agent.models.generate_content(
                model=self.model_name, contents=types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                        ),
                        types.Part(text=VIDEO_PROMPT)
                    ]
                ),
                config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StructuredVideoResponse, # Pass the Pydantic class directly
                ),
            )
        else:
            myfile = self.agent.files.upload(file=video_path)
            time.sleep(10)
            response = self.agent.models.generate_content(
                model=self.model_name, contents=[myfile, VIDEO_PROMPT],
                config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StructuredVideoResponse, # Pass the Pydantic class directly
                ),
            )
        end_time = time.time()
        
        parsed = StructuredVideoResponse.model_validate_json(response.text)
        segments = parsed.segments
        
        inference_cost = (response.usage_metadata.prompt_token_count*self.pricing_dict["input_cost_per_1Mtks"] + response.usage_metadata.candidates_token_count*self.pricing_dict["output_cost_per_1Mtks"])/1000000
        inference_time = end_time - start_time
        return segments, inference_cost, inference_time

if __name__ == "__main__":
    frame_path = "data/frames_fps1/clipped_0_11_cart_v1_8_214.jpg"
    # frame_path = "data/demo.jpg"
    # agent = AgentOpenAI(model_name="gpt-5.4", api_key=OPENAI_API_KEY)
    # agent = AgentClaude(model_name="claude-haiku-4-5-20251001", api_key=CLAUDE_API_KEY)
    agent = AgentGemini(model_name="gemini-3.1-pro-preview", api_key=GEMINI_API_KEY)
    
    # action, inference_cost, inference_time = agent.inference_one_frame(frame_path)
    # print(action, inference_cost, inference_time)

    video_path = "data/reconstructed_videos/clipped_0_11_cart_v1.mp4"
    segments, inference_cost, inference_time = agent.inference_one_video(video_path)
    segments = [(seg.action, seg.start_second) for seg in segments]
    print(segments, inference_cost, inference_time)
