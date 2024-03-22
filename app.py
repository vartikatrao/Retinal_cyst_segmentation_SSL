import pickle
from flask import Flask, render_template
import torch 
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend (for PNGs) instead of the default interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import segmentation_models_pytorch as smp


app = Flask(__name__,template_folder='Templates')
from flask import request
import time 



@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('Predict_image.html')

model=smp.Unet(encoder_name='efficientnet-b2',
               encoder_weights="imagenet",
               decoder_use_batchnorm=True,
               in_channels=1,
               classes=2,
               activation='softmax'
              )
model.load_state_dict(torch.load('models/model_100.pt', map_location=torch.device('cpu')))

@app.route('/predict_results', methods=['GET', 'POST'])
def predict_results():
    if request.method == 'POST':
        # Load the uploaded files
        file = request.files['fileInput']
        np_array = np.load(file)
        plt.imshow(np_array, cmap='gray')
        plt.axis('off')
        plt.savefig('input.png')
        plt.close()
        with open('input.png', 'rb') as f:
            input_data= base64.b64encode(f.read()).decode('utf-8')
        print(np_array.shape)
        img_tensor = torch.from_numpy(np_array).float()
        print(img_tensor.shape)
        img_tensor = img_tensor.unsqueeze(0)
        print(img_tensor.shape)
        img_tensor = img_tensor.unsqueeze(0)
        print(img_tensor.shape)
        with torch.no_grad():
            output = model(img_tensor)
            print("after model", output.shape)
            #output.shape= torch.Size([1, 2, 256, 512])
            #get max of the two channels
            output= torch.max(output, 1)[1]
            print("after max", output.shape)
            output= torch.squeeze(output)
            plt.imshow(output, cmap='gray')
            plt.axis('off')
            plt.savefig('output.png')
            plt.close()
            with open('output.png', 'rb') as f:
                output_data= base64.b64encode(f.read()).decode('utf-8')
        
        # Overlay segmentation output on input
        overlay = np_array.copy()
        overlay[output == 1] = 255  # Set segmentation output to white
        plt.imshow(overlay, cmap='gray')
        plt.axis('off')
        plt.savefig('overlay.png')
        plt.close()
        with open('overlay.png', 'rb') as f:
            overlay_data = base64.b64encode(f.read()).decode('utf-8')
        
        print(output)

        return render_template('results.html', plot_data= output_data, input_data= input_data, overlay_data= overlay_data)
    else:
        return render_template('results.html')



if __name__ == '__main__':
    app.run(debug=True)
