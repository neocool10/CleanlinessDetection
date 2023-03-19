from django.shortcuts import render, HttpResponse, redirect
from home.models import Contact , Plus
from django.contrib import messages 
from django.contrib.auth.models import User 
from django.contrib.auth  import authenticate,  login, logout
from blog.models import Post
from django.core.files.storage import FileSystemStorage

from IPython.display import display, Javascript, Image
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import html
import time
import matplotlib.pyplot as plt
import os

from django.http import HttpResponse
from django.shortcuts import render
from .models import *
from django.http import StreamingHttpResponse
import cv2
import threading
import gzip


def home(request): 
    if os.path.exists("media/testimg.jpg"):
            os.remove("media/testimg.jpg")
    else:
        print("The file does not exist")
    if os.path.exists("static/labeledimg.jpg"):
            os.remove("static/labeledimg.jpg")
    else:
        print("The file does not exist")
    return render(request, "home/home.html")



def search(request):
    query=request.GET['query']
    if len(query)>78:
        allPosts=Post.objects.none()
    else:
        allPostsTitle= Post.objects.filter(title__icontains=query)
        allPostsAuthor= Post.objects.filter(author__icontains=query)
        allPostsContent =Post.objects.filter(content__icontains=query)
        allPosts=  allPostsTitle.union(allPostsContent, allPostsAuthor)
    if allPosts.count()==0:
        messages.warning(request, "No search results found. Please refine your query.")
    params={'allPosts': allPosts, 'query': query}
    return render(request, 'home/search.html', params)

def handleSignUp(request):
    if request.method=="POST":
        # Get the post parameters
        username=request.POST['username']
        email=request.POST['email']
        fname=request.POST['fname']
        lname=request.POST['lname']
        pass1=request.POST['pass1']
        pass2=request.POST['pass2']

        # check for errorneous input
        if len(username)>10:
            messages.error(request, " Your user name must be under 10 characters")
            return redirect('home')

        if not username.isalnum():
            messages.error(request, " User name should only contain letters and numbers")
            return redirect('home')
        if (pass1!= pass2):
             messages.error(request, " Passwords do not match")
             return redirect('home')
        
        # Create the user
        myuser = User.objects.create_user(username, email, pass1)
        myuser.first_name= fname
        myuser.last_name= lname
        myuser.save()
        messages.success(request, " Your profile has been successfully created")
        return redirect('home')

    else:
        return HttpResponse("404 - Not found")


def handeLogin(request):
    if request.method=="POST":
        # Get the post parameters
        loginusername=request.POST['loginusername']
        loginpassword=request.POST['loginpassword']

        user=authenticate(username= loginusername, password= loginpassword)
        if user is not None:
            login(request, user)
            messages.success(request, "Successfully Logged In")
            return redirect("home")
        else:
            messages.error(request, "Invalid credentials! Please try again")
            return redirect("home")

    return HttpResponse("404- Not found")
   

    return HttpResponse("login")

def handelLogout(request):
    logout(request)
    messages.success(request, "Successfully logged out")
    return redirect('home')


def about(request): 
    return render(request, "home/about.html")



def contact(request):
    if request.method == 'POST' and request.FILES['vid']:
        video = request.FILES['vid']
        if os.path.exists("media/testvid.mp4"):
            os.remove("media/testvid.mp4")
        else:
            print("The file does not exist")
        # if os.path.exists("static/labeledimg.jpg"):
        #     os.remove("static/labeledimg.jpg")
        # else:
            print("The file does not exist")
        fss = FileSystemStorage()
        file = fss.save('testvid.mp4', video)
        file_url = fss.url(file)

        #network, class_names, class_colors = darknet.load_network("darknet/cfg/yolov4-csp.cfg", "darknet/cfg/coco.data", "darknet/yolov4-csp.weights")
        network, class_names, class_colors = darknet.load_network("darknet/cfg/cleanliness.cfg", "darknet/cfg/cleanobj.data", "darknet/cleanliness_final.weights")
        cap = cv2.VideoCapture('media/testvid.mp4')
        ret, frame = cap.read()
        height,width,layers=frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter('output.avi', fourcc, 10.0, (width, height) )
        while (cap.isOpened()):
            ret, frame = cap.read()
            #output_img = frame_photo(frame,width,height,network,class_names)
            output_img = framephoto(frame)
            video.write(output_img)
        cap.release()
        cv2.destroyAllWindows()
        video.release()
        
        # ...
        # image object ist contained in canvas
        
        
        context = {
        'canvas': canvas,
    }

        messages.success(request, "Your image has been successfully uploaded")
        return render(request,response, "home/contact.html", {'file_url': file_url})
    return render(request, 'home/contact.html')

def plus(request): 
    if request.method=="POST" and request.FILES['plus']:
        plus = request.FILES['plus']
        if os.path.exists("media/testimg.jpg"):
            os.remove("media/testimg.jpg")
        else:
            print("The file does not exist")
        if os.path.exists("static/labeledimg.jpg"):
            os.remove("static/labeledimg.jpg")
        else:
            print("The file does not exist")
        fss = FileSystemStorage()
        
        file = fss.save('testimg.jpg', plus)
        file_url = fss.url(file)
        #network, class_names, class_colors = darknet.load_network("darknet/cfg/yolov4-csp.cfg", "darknet/cfg/coco.data", "darknet/yolov4-csp.weights")
        network, class_names, class_colors = darknet.load_network("darknet/cfg/cleanliness.cfg", "darknet/cfg/cleanobj.data", "darknet/cleanliness_final.weights")
        width = darknet.network_width(network)
        height = darknet.network_height(network)
        
        # run test on person.jpg image that comes with repository
        image = cv2.imread('media/testimg.jpg')
        detections, width_ratio, height_ratio = darknet_helper(image, width, height, network, class_names)
        
        for label, confidence, bbox in detections:
            left, top, right, bottom = darknet.bbox2points(bbox)
            left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
            cv2.rectangle(image, (left, top), (right, bottom), class_colors[label], 2)
            cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                                (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                class_colors[label], 2)
        cv2.imwrite('static/labeledimg.jpg',image)
        if detections!=[]:
            for i in range(len(detections)):
                print(detections)
                messages.success(request, detections[i][0]+" Detected with "+detections[i][1]+" Accuracy.")
        

        #images = image.objects.all()
        #messages.success(request, "Your image has been successfully uploaded")
        return render(request, "home/plus.html", {'file_url': file_url})
    return render(request, "home/plus.html")



# import darknet functions to perform object detections
from darknet import darknet
# load in our YOLOv4 architecture network

# darknet helper function to run detection on image
def darknet_helper(img, width, height, network, class_names):
  darknet_image = darknet.make_image(width, height, 3)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(img_rgb, (width, height),
                              interpolation=cv2.INTER_LINEAR)

  # get image ratios to convert bounding boxes to proper size
  img_height, img_width, _ = img.shape
  width_ratio = img_width/width
  height_ratio = img_height/height

  # run model on darknet style image to get detections
  darknet.copy_image_from_bytes(darknet_image, img_resized.tobytes())
  detections = darknet.detect_image(network, class_names, darknet_image)
  darknet.free_image(darknet_image)
  return detections, width_ratio, height_ratio






# function to convert the JavaScript object into an OpenCV image

# function to convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream
def bbox_to_bytes(bbox_array):
  """
  Params:
          bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
  Returns:
        bytes: Base64 image byte string
  """
  # convert array into PIL image
  bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
  iobuf = io.BytesIO()
  # format bbox into png for return
  bbox_PIL.save(iobuf, format='png')
  # format return string
  bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))

  return bbox_bytes






def frame_photo(frame,width,height,network,class_names):
  
  # get OpenCV format image
  img = frame
  
  # call our darknet helper on webcam image
  detections, width_ratio, height_ratio = darknet_helper(img, width, height, network, class_names)

  # loop through detections and draw them on webcam image
  for label, confidence, bbox in detections:
    left, top, right, bottom = darknet.bbox2points(bbox)
    left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
    cv2.rectangle(img, (left, top), (right, bottom), darknet.class_colors[label], 2)
    cv2.putText(img, "{} [{:.2f}]".format(label, float(confidence)),
                      (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                      darknet.class_colors[label], 2)
  # save image
  cv2.imwrite('media/images/photo.jpg', img)

  return img

def framephoto(img):
    net = cv2.dnn.readNet("darknet/cleanliness_final.weights", "darknet/cfg/cleanliness.cfg")
    classes = []

    with open("darknet/data/obj.names", "r") as file:
        classes = [line.strip() for line in file.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    colorRed = (0,0,255)
    colorGreen = (0,255,0)

     #Loading Images
    #name = "image.jpg"
    #img = cv2.imread(name)
    height, width, channels = img.shape

    # # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), colorGreen, 3)
            #cv2.putText(img, label, (x, y + 10), cv2.FONT_HERSHEY_PLAIN, 8, colorRed, 8)
            cv2.putText(img, label,(x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorRed, 2)

    #cv2.imshow("Image", img)
    cv2.imwrite("media/images/output.jpg",img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return img



