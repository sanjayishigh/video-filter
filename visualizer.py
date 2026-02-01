import cv2
import numpy as np
import base64
from flask import Blueprint, render_template, request, jsonify

vis_bp = Blueprint('visualizer', __name__)

# Enhanced kernel library
kernels = {
    'Edge Detection': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
    'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    'Box Blur': np.ones((5, 5), dtype=np.float32) / 25,
    'Gaussian Blur': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16,
    'Emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32),
    'Outline': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
    'Sobel X': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    'Sobel Y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
    'Laplacian': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
}

@vis_bp.route('/convolution')
def convolution_page():
    """Serve the convolution visualization page"""
    return render_template('convolution.html')

@vis_bp.route('/analyze_convolution', methods=['POST'])
def analyze():
    """
    Process uploaded image with selected kernel
    Returns base64 encoded input and output images with kernel data
    """
    try:
        # Get uploaded file and kernel selection
        file = request.files.get('image')
        k_name = request.form.get('kernel', 'Edge Detection')
        
        if not file:
            return jsonify({'error': 'No image provided'}), 400
        
        # Read and decode image
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Resize for processing (maintain aspect ratio)
        height, width = img.shape[:2]
        max_dimension = 600
        
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale for convolution
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get kernel
        kernel = kernels.get(k_name, kernels['Edge Detection'])
        
        # Apply convolution
        output = cv2.filter2D(gray, -1, kernel)
        
        # Normalize output for better visualization
        output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
        output = output.astype(np.uint8)
        
        # For edge detection and similar, apply additional processing
        if k_name in ['Edge Detection', 'Sobel X', 'Sobel Y', 'Laplacian', 'Outline']:
            # Threshold for cleaner edges
            _, output = cv2.threshold(output, 50, 255, cv2.THRESH_BINARY)
        
        # Create color-coded version for better visualization
        output_colored = cv2.applyColorMap(output, cv2.COLORMAP_HOT)
        
        # Encode images to base64
        _, buf_in = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, 90])
        _, buf_out = cv2.imencode('.jpg', output_colored, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Prepare response
        response = {
            'input': base64.b64encode(buf_in).decode('utf-8'),
            'output': base64.b64encode(buf_out).decode('utf-8'),
            'kernel': kernel.tolist(),
            'kernel_name': k_name,
            'kernel_size': f"{kernel.shape[0]}x{kernel.shape[1]}",
            'input_size': f"{img.shape[1]}x{img.shape[0]}"
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@vis_bp.route('/get_kernels', methods=['GET'])
def get_kernels():
    """Return list of available kernels with descriptions"""
    kernel_info = {
        'Edge Detection': {
            'description': 'Detects edges by finding areas of high intensity change',
            'use_case': 'Object boundary detection, feature extraction',
            'kernel': kernels['Edge Detection'].tolist()
        },
        'Sharpen': {
            'description': 'Enhances edges and fine details in the image',
            'use_case': 'Image enhancement, detail amplification',
            'kernel': kernels['Sharpen'].tolist()
        },
        'Box Blur': {
            'description': 'Averages pixels in a neighborhood for smoothing',
            'use_case': 'Noise reduction, preprocessing',
            'kernel': kernels['Box Blur'].tolist()
        },
        'Gaussian Blur': {
            'description': 'Weighted averaging with emphasis on center pixels',
            'use_case': 'Natural blur, noise reduction',
            'kernel': kernels['Gaussian Blur'].tolist()
        },
        'Emboss': {
            'description': 'Creates a 3D raised appearance',
            'use_case': 'Artistic effects, texture analysis',
            'kernel': kernels['Emboss'].tolist()
        },
        'Sobel X': {
            'description': 'Detects horizontal edges',
            'use_case': 'Gradient computation, edge direction',
            'kernel': kernels['Sobel X'].tolist()
        },
        'Sobel Y': {
            'description': 'Detects vertical edges',
            'use_case': 'Gradient computation, edge direction',
            'kernel': kernels['Sobel Y'].tolist()
        },
        'Laplacian': {
            'description': 'Second derivative operator for edge detection',
            'use_case': 'Edge detection, zero-crossing analysis',
            'kernel': kernels['Laplacian'].tolist()
        }
    }
    
    return jsonify(kernel_info)

@vis_bp.route('/batch_process', methods=['POST'])
def batch_process():
    """
    Process single image with multiple kernels for comparison
    """
    try:
        file = request.files.get('image')
        
        if not file:
            return jsonify({'error': 'No image provided'}), 400
        
        # Read image
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Resize
        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)
        
        results = {}
        
        # Process with all kernels
        for k_name, kernel in kernels.items():
            output = cv2.filter2D(img, -1, kernel)
            output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
            output = output.astype(np.uint8)
            
            # Encode
            _, buf = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 85])
            results[k_name] = base64.b64encode(buf).decode('utf-8')
        
        # Original image
        _, buf_orig = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        return jsonify({
            'original': base64.b64encode(buf_orig).decode('utf-8'),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500

@vis_bp.route('/step_by_step', methods=['POST'])
def step_by_step():
    """
    Generate step-by-step visualization of convolution on a small patch
    """
    try:
        file = request.files.get('image')
        k_name = request.form.get('kernel', 'Sharpen')
        x = int(request.form.get('x', 50))
        y = int(request.form.get('y', 50))
        
        # Read image
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 200))
        
        kernel = kernels.get(k_name, kernels['Sharpen'])
        k_h, k_w = kernel.shape
        pad = k_h // 2
        
        # Extract patch
        patch = img[max(0, y-pad):y+pad+1, max(0, x-pad):x+pad+1]
        
        # Element-wise multiplication
        if patch.shape == kernel.shape:
            multiplied = patch * kernel
            result = np.sum(multiplied)
            
            return jsonify({
                'patch': patch.tolist(),
                'kernel': kernel.tolist(),
                'multiplied': multiplied.tolist(),
                'result': float(result)
            })
        else:
            return jsonify({'error': 'Patch extraction failed'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Step-by-step failed: {str(e)}'}), 500