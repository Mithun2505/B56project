from django.shortcuts import render
from django.views.generic import TemplateView
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
import cv2 as cv
import numpy as np
#from tkinter import Tk
#Tk().withdraw()
#from tkinter.filedialog import askopenfilename
from cancerTB.models import *


# Create your views here.

class HomeView(TemplateView):
    template_name = 'cancerTB/home.html'

    def post(self, request):

        if 'uploadfile' in request.POST:
            test = request.POST.get('fileupload')
            print(test, 'filename')
            test = '/media/sf_New_folder/dataset_final/test_data/1/' + test
            print(test)
            image = cv.imread(test)
            img = (np.expand_dims(image,0))
            answer = self.run_model(img)
            print(answer)
            answer = int(answer[0][0])
            #print(answer, 'Answer')
            if answer == 1:
                cancerAns = 'Positive'
            else:
                cancerAns = 'Negative'
            return render(request, 'cancerTB/home.html', {'cancerAns': cancerAns})

    def run_model(self, image):
        #print('test')
        model = tf.keras.models.load_model('/media/sf_New_folder/final_cancer.h5')
        out = model.predict(image)
        return out

