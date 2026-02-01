import cv2
import numpy as np
from flask import Blueprint, render_template, Response, request, jsonify
from camera import cam_instance

filters_bp = Blueprint('filters', __name__)

# State
filter_config = {
    'mode': 'none',
    'mirror': True,  # Default to mirrored (selfie mode)
    'split': False
}

# --- FILTERS ---
def apply_vivid_enhance(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.medianBlur(a, 5); b = cv2.medianBlur(b, 5)
    l = cv2.medianBlur(l, 3)
    l = cv2.bilateralFilter(l, d=5, sigmaColor=50, sigmaSpace=50)
    gaussian_l = cv2.GaussianBlur(l, (5, 5), 2.0)
    l = cv2.addWeighted(l, 1.5, gaussian_l, -0.5, 0)
    merged = cv2.merge((l, a, b))
    bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 1.45)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def apply_studio(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def apply_glow(frame):
    blur = cv2.GaussianBlur(frame, (0, 0), 3)
    return cv2.addWeighted(frame, 0.7, blur, 0.3, 0)

kernels = {
    'sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    'emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
}

def process_frame(frame):
    mode = filter_config['mode']
    if mode == 'vivid': return apply_vivid_enhance(frame)
    elif mode == 'studio': return apply_studio(frame)
    elif mode == 'glow': return apply_glow(frame)
    elif mode in kernels: return cv2.filter2D(frame, -1, kernels[mode])
    return frame

def gen_frames():
    while True:
        success, frame = cam_instance.get_frame()
        if not success: break
        
        # 1. MIRROR LOGIC
        if filter_config['mirror']:
            frame = cv2.flip(frame, 1)

        # Process Filter
        output = process_frame(frame)
        
        # 2. SPLIT VIEW LOGIC
        if filter_config['split'] and filter_config['mode'] != 'none':
            h, w = frame.shape[:2]
            mid = w // 2
            
            # Ensure output matches input size/channels (crucial for edge filters)
            if output.shape[:2] != (h, w):
                output = cv2.resize(output, (w, h))
            if len(output.shape) == 2: # Convert grayscale to BGR
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

            # Create Split: Left = Original, Right = Filtered
            # We cut the original frame in half and the filtered frame in half
            combined = np.hstack((frame[:, :mid, :], output[:, mid:, :]))
            
            # Draw Divider Line
            cv2.line(combined, (mid, 0), (mid, h), (255, 107, 53), 2)
            output = combined

        ret, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ROUTES
@filters_bp.route('/')
def index(): return render_template('index.html')

@filters_bp.route('/video_feed')
def video_feed(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@filters_bp.route('/set_mode/<mode>', methods=['POST'])
def set_mode(mode):
    filter_config['mode'] = mode
    return jsonify({'status': 'success'})

@filters_bp.route('/toggle/<feature>', methods=['POST'])
def toggle(feature):
    if feature in filter_config:
        filter_config[feature] = not filter_config[feature]
    return jsonify({'state': filter_config[feature]})