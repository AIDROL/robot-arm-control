"""Actual OpenCV detections from viewport pixels, displayed inside Isaac.

No projected object truth, cached detections or pose fallback. Display only.
"""
import ctypes
import json
from pathlib import Path
import time

import cv2
import numpy as np


def annotate(rgba, labels):
    """Return RGBA annotations and image-derived IDs/corners, without changing input."""
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    canvas = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    records = []
    for quad, marker in zip(corners, [] if ids is None else ids.flatten()):
        marker = int(marker)
        points = quad.reshape(4, 2)
        pixel = np.rint(points).astype(np.int32)
        cv2.polylines(canvas, [pixel], True, (70, 240, 70), 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(pixel):
            color = (50, 70, 255) if i == 0 else (0, 220, 255)
            cv2.circle(canvas, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
            dx, dy = [(-22, -8), (6, -8), (6, 18), (-22, 18)][i]
            cv2.putText(canvas, str(i), (int(x)+dx, int(y)+dy), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
        label = f'ID {marker}' + (' | '+labels[marker] if marker in labels else ' | unregistered')
        x, y = int(pixel[:, 0].min()), max(22, int(pixel[:, 1].min())-22)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .55, 1)
        x = max(0, min(x, canvas.shape[1]-w-8))
        cv2.rectangle(canvas, (x, y-h-5), (min(canvas.shape[1]-1, x+w+8), y+5), (24, 28, 30), -1)
        cv2.putText(canvas, label, (x+4, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (100, 255, 150), 1, cv2.LINE_AA)
        records.append({'id': marker, 'label': labels.get(marker), 'corners_px': points.tolist()})
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGBA), records


def component_labels(config, bindings):
    public = {b['backend']: b['id'] for b in bindings}
    result = {20: 'FAT / top'}
    for marker, component in enumerate(config['components']):
        label = public.get(component['id'], component['id'])
        result[marker] = label
        result[marker+10] = label+' / side'
    return result


class ArucoView:
    def __init__(self, viewport, labels, output=None):
        import omni.ui as ui
        from omni.kit.viewport.utility import capture_viewport_to_buffer
        self.viewport, self.labels = viewport, labels
        self.capture_buffer = capture_viewport_to_buffer
        self.output = Path(output) if output else None
        self.pending, self.last_capture, self.capture = False, -1., None
        self.frames, self.saved, self.closed = 0, False, False
        self.provider = ui.ByteImageProvider()
        self.window = ui.Window('OpenCV ArUco | active Isaac camera', width=920, height=590)
        with self.window.frame:
            with ui.VStack(spacing=5):
                ui.Label('LIVE VIEWPORT PIXELS -> OpenCV DICT_4X4_50 | green quad | corners 0-3 | ID / component', height=24)
                self.status = ui.Label('Waiting for camera pixels...', height=24)
                ui.ImageWithProvider(self.provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                ui.Label('Image diagnostics only. No marker visible = no label. No object-truth projection.', height=24)
        self.error = None

    def update(self, source_time_s):
        wall = time.monotonic()
        if self.closed or not self.window.visible or self.pending or wall-self.last_capture < .2:
            return
        self.pending = True
        self.last_capture = wall
        camera = str(self.viewport.camera_path)
        def captured(buffer, size, width, height, pixel_format):
            try:
                if self.closed:
                    return
                # Kit delivers a PyCapsule. Copy while the callback owns it.
                if size != width*height*4 or 'RGBA8' not in str(pixel_format).upper():
                    raise ValueError(f'Unsupported viewport pixels: {pixel_format}, size={size}')
                get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
                get_pointer.restype = ctypes.c_void_p
                get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
                pointer = get_pointer(buffer, None)
                pixels = np.ctypeslib.as_array((ctypes.c_uint8*size).from_address(pointer)).copy().reshape(height,width,4)
                annotated, records = annotate(pixels, self.labels)
                self.provider.set_data_array(np.ascontiguousarray(annotated), [width,height])
                self.frames += 1
                ids = ', '.join(str(r['id'])+'='+str(r['label'] or 'unregistered') for r in records) or 'none'
                self.status.text = f't={source_time_s:.2f}s | {camera} | {width}x{height} | detected: {ids}'
                report = {'source':'live_viewport_pixels', 'detector':'OpenCV ArucoDetector / DICT_4X4_50',
                          'source_time_s':source_time_s,'camera':camera,'width':width,'height':height,
                          'detections':records,'frames_processed':self.frames}
                if self.output:
                    self.output.mkdir(parents=True, exist_ok=True)
                    temporary=self.output/'aruco-current.tmp'
                    temporary.write_text(json.dumps(report,allow_nan=False,indent=2))
                    temporary.replace(self.output/'aruco-current.json')
                    if records and not self.saved:
                        cv2.imwrite(str(self.output/'aruco-input.png'),cv2.cvtColor(pixels,cv2.COLOR_RGBA2BGRA))
                        cv2.imwrite(str(self.output/'aruco-detections.png'),cv2.cvtColor(annotated,cv2.COLOR_RGBA2BGRA))
                        (self.output/'aruco-first-detections.json').write_text(json.dumps(report,indent=2))
                        self.saved=True
                self.error=None
            except Exception as exc:
                self.error=str(exc)
                self.status.text='OpenCV camera unavailable: '+self.error
                print('ARUCO_VIEW_ERROR',self.error,flush=True)
            finally:
                self.pending=False
        self.capture=self.capture_buffer(self.viewport,captured)

    def close(self):
        self.closed=True
        self.window.visible=False
