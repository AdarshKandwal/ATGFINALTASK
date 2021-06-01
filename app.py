import sys
import io
import os
import time
import flask
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML
import warnings
import cv2  
import shutil

import time
from demo import make_animation
from skimage import img_as_ubyte


app = Flask(__name__)


#1. Handling all the requests 

@app.after_request
def add_header(response):
 
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'

    return response
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/indexcatdog', methods=['GET'])
def project():

    return render_template('images.html')

@app.route('/predictcatdog', methods=['GET', 'POST'])
def upload_catdog():
    if request.method == 'POST':
        # Get the file from post request
        files = flask.request.files.getlist("file[]")
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        count =0
        for file in files:
            if count==0:
                file.save(os.path.join(basepath,'uploads',"Source.mp4"))
                SourcePath=os.path.join(basepath,'uploads',"Source.mp4")   
    # Make prediction
        shutil.copy(SourcePath,"static/")
        base_image_path="input_voldmort/voldemort3"
        #base_image_path="input_voldmort/voldemort3"
        deep_fake(base_image_path,SourcePath)

        time.sleep(5)
        
        return render_template('result.html')
    return None   




def display(source, driving, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

    ims = []
    for i in range(len(driving)):
        cols = [source]
        cols.append(driving[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    writervideo = animation.FFMpegWriter(fps=60)
    ani.save('static/final_graph.mp4', writer=writervideo)
    plt.close()
    return ani
    


#2 . Now Taking this face and Deep Faking it onto other video .

def deep_fake(source_image,driving_video):
    source_image = imageio.imread(source_image)
    driving_video = imageio.mimread(driving_video)
    # Resize image and video to 256x256

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    
    from demo import load_checkpoints
    generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml', 
                            checkpoint_path='D:/Desktop4/vox-cpk.pth.tar',cpu=True)
    
    Z
    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True,cpu=True)
    imageio.mimsave('static/generated.mp4', [img_as_ubyte(frame) for frame in predictions])
    generated_video = imageio.mimread('static/generated.mp4')
    generated_video = [resize(frame, (256, 256))[..., :3] for frame in generated_video]

    HTML(display(source_image ,driving_video,generated_video).to_html5_video())
    #video can be downloaded from /content folder




if __name__ == '__main__':
    app.run(debug=True)     