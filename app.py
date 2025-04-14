from flask import Flask, render_template, request, send_file
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def convert_to_sketch(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image
    inverted_image = cv2.bitwise_not(gray_image)
    
    # Apply Gaussian blur to the inverted image
    blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)
    
    # Invert the blurred image
    inverted_blurred = cv2.bitwise_not(blurred)
    
    # Create the pencil sketch by blending the grayscale image with the inverted blurred image
    sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
    
    # Save the sketch image
    sketch_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sketch.jpg')
    cv2.imwrite(sketch_path, sketch)
    return sketch_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file selected')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            sketch_path = convert_to_sketch(filepath)
            return render_template('index.html', 
                                 original_image=filename,
                                 sketch_image='sketch.jpg')
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)
