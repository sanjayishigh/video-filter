import cv2
import numpy as np
from flask import Blueprint, render_template, Response, request, jsonify
from camera import cam_instance

filters_bp = Blueprint('filters', __name__)

# State
filter_config = {
    'mode': 'none',
    'mirror': True,
    'split': False
}

# --- STANDARD KERNELS ---
kernels = {
    'sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    'emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
    'outline': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
}

# --- CUSTOM FILTER FUNCTIONS ---

def apply_custom_laplacian(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    convolved = cv2.filter2D(gray, cv2.CV_64F, lap_kernel)
    convolved = np.absolute(convolved) * 3.5
    convolved = np.clip(convolved, 0, 255).astype(np.uint8)
    return cv2.cvtColor(convolved, cv2.COLOR_GRAY2BGR)

def apply_glow(frame):
    gaussian_kernel = np.array([
        [1, 4,  6,  4,  1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4,  6,  4,  1]
    ], dtype=np.float32) / 256.0
    blur = cv2.filter2D(frame, -1, gaussian_kernel)
    blur = cv2.filter2D(blur, -1, gaussian_kernel)
    return cv2.addWeighted(frame, 0.7, blur, 0.3, 0)

def apply_lowpass(frame):
    lowpass_kernel = np.array([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ], dtype=np.float32) / 25.0
    return cv2.filter2D(frame, -1, lowpass_kernel)

def apply_soft_enhance(frame):
    """
    Soft Mode - Bilateral Filter (Optimized)
    Edge-preserving smoothing that reduces noise while keeping sharp boundaries.
    Optimized parameters for natural skin-like smoothing.
    """
    # Bilateral filter: smooths while preserving edges
    # d=7 (diameter), sigmaColor=50, sigmaSpace=50
    smoothed = cv2.bilateralFilter(frame, 7, 50, 50)
    
    # Subtle saturation boost
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.2, 0, 255).astype(np.uint8)
    v = np.clip(v * 1.05, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def process_frame(frame):
    mode = filter_config['mode']
    if mode == 'soft': return apply_soft_enhance(frame)
    elif mode == 'lowpass': return apply_lowpass(frame)
    elif mode == 'glow': return apply_glow(frame) 
    elif mode == 'laplacian': return apply_custom_laplacian(frame)
    elif mode in kernels: return cv2.filter2D(frame, -1, kernels[mode])
    return frame

# --- GENERATOR & ROUTES ---
def gen_frames():
    while True:
        success, frame = cam_instance.get_frame()
        if not success: break
        
        if filter_config['mirror']:
            frame = cv2.flip(frame, 1)

        output = process_frame(frame)
        
        if filter_config['split'] and filter_config['mode'] != 'none':
            h, w = frame.shape[:2]
            mid = w // 2
            if output.shape[:2] != (h, w):
                output = cv2.resize(output, (w, h))
            if len(output.shape) == 2:
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

            combined = np.hstack((frame[:, :mid, :], output[:, mid:, :]))
            cv2.line(combined, (mid, 0), (mid, h), (255, 107, 53), 2)
            output = combined

        ret, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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