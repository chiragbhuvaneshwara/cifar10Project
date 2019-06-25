import flask
import werkzeug
import os
import scipy.misc
from models import *
from matplotlib.pyplot import imread

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

def loadModelAndTest():
    
    global secure_filename
    #Reading the image file from the path it was saved in previously.
    img = imread(os.path.join(app.root_path, secure_filename))
    img = torch.tensor(img, device=device).float()
    img = img.unsqueeze(0)
    print('################################################')
    img = img.permute(0,3,1,2) 
    shape = np.array(img.size()) 
    reqd = np.array([1,3,32,32])
    
    if (np.array_equal(shape, reqd)):
        model= modelClass(num_classes, fine_tune, pretrained)
        model.load_state_dict(torch.load(path+'TransferLearning_models/'+ modelName +'.ckpt', map_location=device))
        model.to(device)
        model.eval()                
        images = torch.cat((img,img),0)
        outputs = model(images)
        _, predicted_classes = torch.max(outputs.data, 1)
        predicted_class = predicted_classes[0]
        predicted_class = labels[int(predicted_class)]
        
        return flask.render_template(template_name_or_list="prediction_result.html", predicted_class=predicted_class)

    else:
        # If the image dimensions do not match the CIFAR10 specifications, then an HTML page is rendered to show the problem.
        return flask.render_template(template_name_or_list="error.html", img_shape=img.shape)

def upload_image():
   
    global secure_filename
    if flask.request.method == "POST":#Checking of the HTTP method initiating the request is POST.
        img_file = flask.request.files["image_file"]#Getting the file name to get uploaded.
        secure_filename = werkzeug.secure_filename(img_file.filename)#Getting a secure file name. It is a good practice to use it.
        img_path = os.path.join(app.root_path, secure_filename)#Preparing the full path under which the image will get saved.
        img_file.save(img_path)#Saving the image in the specified path.
        print("Image uploaded successfully.")
      
        return flask.redirect(flask.url_for(endpoint="predict"))
    return "Image upload failed."

def redirect_upload():

    return flask.render_template(template_name_or_list="upload_image.html")

# Parameters for the model
path = '../'
modelClass = ResNetModel
modelName = 'resnetModel'
fine_tune = False
pretrained = True
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = 10

# New web app
app = flask.Flask("CIFAR10_Flask_Web_App")

# Adding urls and associated functions
app.add_url_rule(rule="/predict/", endpoint="predict", view_func=loadModelAndTest)
app.add_url_rule(rule="/upload/", endpoint="upload", view_func=upload_image, methods=["POST"])
app.add_url_rule(rule="/", endpoint="homepage", view_func=redirect_upload)

if __name__ == "__main__":
    app.run(host="localhost", port=7777, debug=True)