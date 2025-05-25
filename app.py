from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import torch
import torch.nn as nn
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Down-sampling layers
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # Output: 64x64
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # Bottleneck
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),  # Output: 8x8
            nn.ReLU(),

            # Up-sampling layers
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Extra upsampling layer to get back to 128x128
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # Output: 128x128
            nn.Tanh()  # Output layer with tanh for [-1, 1] pixel range
        )

    def forward(self, x):
        return self.main(x)

# Instantiate the models
G = Generator()
F = Generator()

# Load the saved model weights
G.load_state_dict(torch.load('generator_G.pth', map_location=device))
F.load_state_dict(torch.load('generator_F.pth', map_location=device))

G.to(device).eval()  # Set to eval mode for inference
F.to(device).eval()  # Set to eval mode for inference

# Define image transformations (resize, tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Assuming images were normalized during training
])

# Function to convert tensor to PIL Image
def tensor_to_image(tensor):
    tensor = tensor.cpu().clone().detach()  # Clone and detach the tensor
    tensor = (tensor + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    tensor = tensor.squeeze(0)  # Remove batch dimension
    image = transforms.ToPILImage()(tensor)
    return image

# Set upload folder
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Helper function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle file uploads for sketch -> real image
@app.route('/upload_sketch', methods=['POST'])
def upload_sketch():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        img = Image.open(filepath).convert('L')  # Convert to grayscale
        img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

        # Generate real image from sketch using G
        with torch.no_grad():
            generated_real = G(img_tensor)

        # Convert tensor back to image
        generated_real_img = tensor_to_image(generated_real)

        # Save the generated image
        output_filename = 'generated_real_' + filename
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        generated_real_img.save(output_filepath)

        return render_template('result.html', original=filename, generated=output_filename)

# Route to handle file uploads for real image -> sketch
@app.route('/upload_real', methods=['POST'])
def upload_real():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        img = Image.open(filepath).convert('L')  # Convert to grayscale
        img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

        # Generate sketch from real image using F
        with torch.no_grad():
            generated_sketch = F(img_tensor)

        # Convert tensor back to image
        generated_sketch_img = tensor_to_image(generated_sketch)

        # Save the generated image
        output_filename = 'generated_sketch_' + filename
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        generated_sketch_img.save(output_filepath)

        return render_template('result.html', original=filename, generated=output_filename)

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
