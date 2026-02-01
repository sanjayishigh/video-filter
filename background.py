import cv2
import numpy as np
from flask import Blueprint, render_template, Response, request, jsonify
from ultralytics import YOLO
from camera import cam_instance

bg_bp = Blueprint('background', __name__)

bg_config = {
    'bg_mode': 'blur',
    'glare_reduce': False,
    'blur_amount': 21 # Adjusted for lower res
}

state = { 'prev_mask': None }

try:
    seg_model = YOLO('yolov8n-seg.pt')
    HAS_SEG = True
except:
    HAS_SEG = False

def process_bg(frame):
    if not HAS_SEG: return frame

    # 1. OPTIMIZATION: Process at tiny resolution
    # Using 160px is extremely fast on CPU
    h, w = frame.shape[:2]
    
    # 2. Run YOLO with forced small size
    # imgsz=160 is the key to high FPS on CPU
    results = seg_model(frame, classes=0, verbose=False, conf=0.4, imgsz=160)
    
    if not results or not results[0].masks:
        return frame

    # 3. Get Mask & Resize to match Frame
    # We let YOLO do the heavy lifting, we just resize the output mask
    mask = results[0].masks.data[0].cpu().numpy()
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 4. Temporal Stability (Low cost blend)
    if state['prev_mask'] is not None:
         mask = cv2.addWeighted(mask, 0.6, state['prev_mask'], 0.4, 0)
    state['prev_mask'] = mask

    # 5. Blur Background
    # Scale blur amount based on new resolution
    k = bg_config['blur_amount']
    if k % 2 == 0: k += 1
    bg = cv2.GaussianBlur(frame, (k, k), 0)

    # 6. Composite
    # Convert mask to 3 channels
    mask_3d = np.dstack([mask] * 3)
    
    # Fast numpy math
    fg = (frame * mask_3d).astype(np.uint8)
    bg_part = (bg * (1 - mask_3d)).astype(np.uint8)
    final = cv2.add(fg, bg_part)

    # Anti-Glare (Optional)
    if bg_config['glare_reduce']:
        # Simple Gamma - fast LUT
        table = np.array([((i / 255.0) ** 1.3) * 255 for i in np.arange(0, 256)]).astype("uint8")
        final = cv2.LUT(final, table)

    return final

def gen_bg_frames():
    while True:
        success, frame = cam_instance.get_frame()
        if not success: break
        output = process_bg(frame)
        # Quality 80 for speed
        ret, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ROUTES
@bg_bp.route('/background')
def background_page(): return render_template('background.html')

@bg_bp.route('/bg_feed')
def bg_feed(): return Response(gen_bg_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@bg_bp.route('/toggle_glare', methods=['POST'])
def toggle_glare():
    bg_config['glare_reduce'] = not bg_config['glare_reduce']
    return jsonify({'status': 'ok'})

@bg_bp.route('/set_blur/<int:value>', methods=['POST'])
def set_blur(value):
    if value % 2 == 0: value += 1
    bg_config['blur_amount'] = value
    return jsonify({'status': 'ok'})

@bg_bp.route('/get_fps', methods=['GET'])
def get_fps():
    return jsonify({'fps': int(cam_instance.get_fps())})