import os, re, math
import cv2
import yt_dlp

from pathlib import Path
from moviepy import VideoFileClip

def extract_gif_from_video(video_path, start_time, end_time, output_gif_path="", fps=10, width=480):
    clip = VideoFileClip(video_path).subclipped(start_time, end_time)
    resized_clip = clip.resized(width=width) 
    resized_clip.write_gif(output_gif_path, fps=fps, loop=False)
    
def get_video_length(video_path):
    clip = VideoFileClip(video_path)
    return clip.duration

def get_video_fps(video_path):
    clip = VideoFileClip(video_path)
    return clip.fps

def clip_video(video_path, start_time, end_time, output_video_path="", width=None):
    clip = VideoFileClip(video_path).subclipped(start_time, end_time)

    if width is not None:
        clip = clip.resized(width=width)
    clip.write_videofile(output_video_path, codec="libx264")

def get_video_size(video_path):
    clip = VideoFileClip(video_path)
    return clip.size

def clean_filename(title, title_length_limit=None):
    # Only allow alphabets, numbers, and spaces
    cleaned = re.sub(r'[^A-Za-z0-9 ]', '', title)
    cleaned = cleaned.replace(" ", "_")
    if title_length_limit is not None and isinstance(title_length_limit, int):
        min_title_length = min(title_length_limit, len(cleaned))
        return cleaned[:min_title_length]  # Length limit (optional)
    return cleaned

def download_video(
    url,
    download_dir="videos/youtube",
    title_length_limit=100,
    # ffmpeg_path=r"C:\Users\gwyong1\Downloads\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe",
):
    os.makedirs(download_dir, exist_ok=True)

    with yt_dlp.YoutubeDL({
        "quiet": True,
        # "ffmpeg_location": ffmpeg_path,
    }) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get("title", "video")

    safe_title = clean_filename(title, title_length_limit)
    outtmpl = os.path.join(download_dir, f"{safe_title}.%(ext)s")

    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bv*+ba/b",          # 최고 화질 video + 최고 음질 audio, 안 되면 best fallback
        "merge_output_format": "mp4",  # 최종 mp4로 병합
        # "ffmpeg_location": ffmpeg_path,
        "noplaylist": True,
        "quiet": False,
        # "sleep_interval_requests": 5,
        "cookiefile": r"C:\Users\17346\Downloads\www.youtube.com_cookies.txt",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return os.path.join(download_dir, safe_title + ".mp4")

def extract_frames(video_path: Path, target_fps: int = 8, save_1fps: bool = True,
                   target_dir_fps1: Path = Path(r"C:\Users\17346\src\ConstActLoc\data\frames_fps1"), target_dir_fps8: Path = Path(r"C:\Users\17346\src\ConstActLoc\data\frames_fps8")):
    
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[SKIP] Cannot open: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps if original_fps > 0 else 0

    video_name = video_path.stem

    for sec in range(math.floor(duration)):

        success_count = 0

        for i in range(target_fps):

            t = sec + (i + 0.5) / target_fps
            frame_idx = int(round(t * original_fps))

            if frame_idx >= total_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                print(
                    f"[FAILED FRAME] "
                    f"video={video_name}, sec={sec}, "
                    f"frame_idx={frame_idx}"
                )
                continue

            success_count += 1

            filename = f"{video_name}_{sec}_{frame_idx}.jpg"

            # 8 fps 저장
            cv2.imwrite(str(target_dir_fps8 / filename), frame)

            # 1 fps 저장
            if save_1fps and i == target_fps // 2:
                cv2.imwrite(str(target_dir_fps1 / filename), frame)

        # 한 초에 8개 저장 실패한 경우 출력
        if success_count != target_fps:
            print(
                f"[WARNING] {video_name} | sec={sec} | "
                f"saved {success_count}/{target_fps} frames"
            )

    cap.release()