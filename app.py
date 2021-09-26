from flask import Flask,render_template,request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
app=Flask(__name__, template_folder="templates")
model=load_model('rocksidentification.h5')
print("Loaded model from disk")
@app.route('/')
def index():
    return render_template('indexP.html')
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname('__file__')
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64)) 
        x = image.img_to_array(img)
        print(x)
        x = np.expand_dims(x,axis =0)
        print(x)
        preds = model.predict(x)
        print("prediction",preds)
        index = ['blue calcite','limestone' , 'marble' , 'olivine', 'red crystal']
        prediction = index[np.argmax(preds)]
        text = "The prediction of this rock is : " + prediction
    return text
if __name__ == '__main__':
    app.run(debug = False, threaded = True)

