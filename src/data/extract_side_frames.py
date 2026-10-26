import argparse
import os
import subprocess
import sys
from pathlib import Path
def h264_to_mp4(h264_path: Path, mp4_out: str, ffmpeg_exe: str):
    # Raw h264, Nutrition5k side video fixed 30fps
    cmd = [
        ffmpeg_exe,
        "-framerate", "30",
        "-i", str(h264_path),
        "-c", "copy",
        "-y",
        mp4_out
    ]
    proc = subprocess.run(cmd, capture_output=True)
    return proc.returncode == 0
def extract_with_ffmpeg(mp4_path: str, out_pattern: str, frames_per_video: int, skip_edge: float,
                        ffmpeg_exe: str, ffprobe_exe: str):
    probe_cmd = [
        ffprobe_exe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        mp4_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[WARN] ffprobe failed for {mp4_path}, skip")
        return False
    lines = result.stdout.strip().splitlines()
    if len(lines) < 2:
        print(f"[WARN] Cannot read video info {mp4_path}, skip")
        return False
    fps_str, dur_str = lines[0], lines[1]
    try:
        num, den = map(float, fps_str.split("/"))
        fps = num / den
        duration = float(dur_str)
    except Exception:
        print(f"[WARN] Parse video info failed {mp4_path}, skip")
        return False
    start_t = duration * skip_edge
    end_t = duration * (1 - skip_edge)
    extract_dur = end_t - start_t
    if extract_dur <= 0:
        print(f"[WARN] Valid duration too short {mp4_path}, skip")
        return False
    sample_fps = frames_per_video / extract_dur
    fps_filter = f"fps={sample_fps}"
    cmd = [
        ffmpeg_exe,
        "-ss", str(start_t),
        "-i", mp4_path,
        "-t", str(extract_dur),
        "-vf", fps_filter,
        "-frames:v", str(frames_per_video),
        "-q:v", "2",
        "-y",
        out_pattern
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        print(f"[WARN] ffmpeg extract fail {mp4_path}")
        return False
    return True
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--skip-edge", type=float, default=0.08)
    parser.add_argument("--cameras", nargs="+", default=["A", "B", "C", "D"])
    args = parser.parse_args()
    ffmpeg_exe = r"D:\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"
    ffprobe_exe = r"D:\ffmpeg-master-latest-win64-gpl\bin\ffprobe.exe"
    video_root = Path(args.video_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    camera_suffix = [f"camera_{c}.h264" for c in args.cameras]
    discarded_dish_ids = []
    dish_folders = sorted([p for p in video_root.iterdir() if p.is_dir()])
    for dish_dir in dish_folders:
        dish_id = dish_dir.name
        has_any_video = False
        print(f"\nProcessing dish: {dish_id}")
        for suf in camera_suffix:
            vid_path = dish_dir / suf
            if not vid_path.exists():
                print(f"  {vid_path.name} not found, skip")
                continue
            has_any_video = True
            cam_name = suf.replace("camera_", "").replace(".h264", "")
            cam_out = out_dir / dish_id / cam_name
            cam_out.mkdir(parents=True, exist_ok=True)
            out_pattern = str(cam_out / "frame_%03d.jpg")
            # Temporary mp4 stored inside camera output folder
            temp_mp4 = str(cam_out / "temp.mp4")
            ok_convert = h264_to_mp4(vid_path, temp_mp4, ffmpeg_exe)
            if not ok_convert:
                print(f"  {vid_path.name} -> convert h264 failed, skip")
                continue
            ok_extract = extract_with_ffmpeg(temp_mp4, out_pattern,
                                     args.frames_per_video,
                                     args.skip_edge,
                                     ffmpeg_exe, ffprobe_exe)
            if ok_extract:
                print(f"  {vid_path.name} -> extracted OK")
            # Remove temporary mp4 to save disk space
            try:
                os.remove(temp_mp4)
            except:
                pass
        if not has_any_video:
            print(f"  >>> Dish {dish_id}: NO videos found, discard")
            discarded_dish_ids.append(dish_id)
    discard_txt = out_dir / "discarded_dish.txt"
    with open(discard_txt, "w", encoding="utf-8") as f:
        for did in discarded_dish_ids:
            f.write(did + "\n")
    print(f"\nDone! Total discarded dishes: {len(discarded_dish_ids)}, list saved to {discard_txt}")
if __name__ == "__main__":
    main()
