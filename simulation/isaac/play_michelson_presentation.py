"""Play the synchronized replay movie with camera selection and seeking."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np


def chapters(movie, fps, count):
    result = {ord("a"): 0}
    try:
        info = json.loads(movie.with_suffix(".json").read_text())
        events = Path(info["source_folder"]) / "events.jsonl"
        labels = {"FRESH_POST_ASSEMBLY_VISION": ord("f"), "fat: drive": ord("o"), "fat: withdraw": ord("r")}
        for line in events.read_text().splitlines():
            event = json.loads(line)
            if event["stage"] in labels and info["source_time_start_s"] <= event["time_s"] <= info["source_time_end_s"]:
                frame = round((event["time_s"]-info["source_time_start_s"])/info["source_speed"]*fps)
                result[labels[event["stage"]]] = min(max(frame, 0), count-1)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, help="Save a decoded frame and exit")
    parser.add_argument("--at-seconds", type=float, default=0.)
    parser.add_argument("--max-seconds", type=float, help="Bounded playback verification")
    parser.add_argument("--report", type=Path, help="Save playback timing statistics")
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.movie))
    if not capture.isOpened():
        raise RuntimeError("Could not open presentation movie: " + str(args.movie))
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or count < 1:
        raise RuntimeError("Invalid presentation movie timing")
    offset = min(max(round(args.at_seconds * fps), 0), count - 1)
    if args.snapshot:
        capture.set(cv2.CAP_PROP_POS_FRAMES, offset)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Could not decode snapshot")
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.snapshot), frame)
        capture.release()
        return
    title = "Michelson replay | Space pause | arrows seek | 0 all / 1 overview / 2 global / 3 wrist / 4 FAT / 5 optics"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1600, 900)
    crops = {ord("1"): (0, 720, 0, 1280), ord("2"): (0, 360, 1280, 1920),
             ord("3"): (360, 720, 1280, 1920), ord("4"): (720, 1080, 1280, 1920),
             ord("5"): (720, 1080, 0, 1280)}
    selected = None
    chapter_frames = chapters(args.movie, fps, count)
    started = time.monotonic()
    paused = False
    current = -1
    frame = None
    playback_started = time.monotonic()
    first_display = last_display = None
    displayed = skipped = 0
    try:
        while True:
            loop = time.monotonic()
            if args.max_seconds is not None and loop-playback_started >= args.max_seconds:
                break
            target = offset if paused else min(offset + int((loop-started)*fps), count-1)
            if target != current:
                if current >= 0 and target > current+1:
                    skipped += target-current-1
                if current >= 0 and current+1 < target < current+fps:
                    for _ in range(target-current-1):
                        capture.grab()
                elif target != current + 1:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, decoded = capture.read()
                if not ok:
                    raise RuntimeError("Movie decode failed at frame " + str(target))
                frame, current = decoded, target
                first_display = first_display if first_display is not None else time.monotonic()
                last_display = time.monotonic()
                displayed += 1
            if frame is not None:
                display = frame
                if selected is not None:
                    y0, y1, x0, x1 = crops[selected]
                    crop = frame[y0:y1, x0:x1]
                    scale = min(1920/crop.shape[1], 1080/crop.shape[0])
                    width, height = round(crop.shape[1]*scale), round(crop.shape[0]*scale)
                    display = np.full((1080, 1920, 3), 18, np.uint8)
                    x, y = (1920-width)//2, (1080-height)//2
                    display[y:y+height, x:x+width] = cv2.resize(crop, (width, height))
                cv2.imshow(title, display)
            if current == count-1 and not paused:
                paused, offset = True, current
            # Keep each frame on the movie clock; GUI wait overhead must not
            # accumulate into a progressively slower display cadence.
            delay_ms = round(1000/fps) if paused else max(
                1, int(1000*(started+(current-offset+1)/fps-time.monotonic())))
            key = cv2.waitKeyEx(delay_ms)
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                offset, started = (0 if current == count-1 and not paused else current), time.monotonic()
            elif key in (2424832, 65361, 2555904, 65363):
                direction = -1 if key in (2424832, 65361) else 1
                offset = int(np.clip(current + direction*5*fps, 0, count-1))
                started = time.monotonic()
            elif key in chapter_frames or key in (65360, 65367):
                offset = chapter_frames.get(key, 0 if key == 65360 else count-1)
                started = time.monotonic()
            elif key in crops or key == ord("0"):
                selected = None if key == ord("0") else key
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if args.report:
            report = {"frames_displayed": displayed, "frames_skipped": skipped, "movie_fps": fps,
                      "measured_fps": (displayed-1)/max(last_display-first_display, 1e-6) if displayed > 1 else 0.}
            args.report.write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
