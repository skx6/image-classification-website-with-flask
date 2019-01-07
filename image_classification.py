# coding:utf-8
 
from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify, flash
import os
import cv2
import time
 
from datetime import timedelta
from classification.classify import load_model, load_image, classify, import_category_dict

# Load classification model and category dictionary.
model = load_model()
category_dict = import_category_dict('./static/images/categories.txt')

# Picture extension supported.
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp', 'jpeg', 'JPEG'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
# Set static file cache expiration time
app.send_file_max_age_default = timedelta(seconds=1)

@app.route('/', methods=['POST', 'GET'])  # add route
def image_classification():
    basepath = os.path.dirname(__file__)  # current path
    upload_path = os.path.join(basepath, 'static/images','test.jpg')
    if request.method == 'POST':
        if request.form['submit'] == 'upload':
            if len(request.files) == 0:
                return render_template('upload_finish.html', message='Please select a picture file!')
            else:
                f = request.files['picture']
         
                if not (f and allowed_file(f.filename)):
                    # return jsonify({"error": 1001, "msg": "Examine picture extension, only png, PNG, jpg, JPG, or bmp supported."})
                    return render_template('upload_finish.html', message='Examine picture extension, png、PNG、jpg、JPG、bmp support.')
                else:

                    f.save(upload_path)
             
                    # transform image format and name with opencv.
                    img = cv2.imread(upload_path)
                    cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)
             
                    return render_template('upload_finish.html', message='Upload successfully!')

        elif request.form['submit'] == 'classify':
            start_time = time.time()
            image = load_image('./static/images/test.jpg')
            pred, prob = classify(model, image, category_dict)
            pred_time = time.time() - start_time
            top_pred, top_prob= str(pred[0]), prob[0]
            return render_template('upload_finish.html', 
                message="This is {} with probability of {:3f}%, cost {:3f} seconds.".format(top_pred, top_prob*100, pred_time))
 
    return render_template('upload.html')


if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=8080, debug=True)
