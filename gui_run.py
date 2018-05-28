import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog, Checkbutton, Radiobutton
import tkinter.ttk as ttk
import facedetect_mtcnn as mtcnn
def imports():
    global sys
    global cv2
    global np
    global mx
    global pl
    global time
    global pickle
    import sys
#     reload(sys)
#     sys.setdefaultencoding('utf8')
    import mxnet as mx
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import pickle

    import pylab as pl
    from collections import namedtuple
    import time

    global Batch
    Batch = namedtuple('Batch', ['data'])


    # In[2]:
    global MAX_INPUT_DIM
    global prob_thresh
    global nms_thresh
    MAX_INPUT_DIM=5000.0
    prob_thresh = 0.5
    nms_thresh = 0.1

    # In[4]:
    global nms
    def nms(dets, prob_thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= prob_thresh)[0]

            order = order[inds + 1]
        return keep


    # In[5]:
    global sym, arg_params, aux_params
    global all_layers
    sym, arg_params, aux_params = mx.model.load_checkpoint('hr101',0)
    all_layers = sym.get_internals()
    global meta_file
    meta_file = open('meta.pkl', 'rb')
    global clusters
    clusters = pickle.load(meta_file, encoding='latin1')
    global averageImage
    averageImage = pickle.load(meta_file, encoding='latin1')

    meta_file.close()
    
    # In[6]:
    global clusters_h
    global clusters_w
    global normal_idx
    clusters_h = clusters[:,3] - clusters[:,1] + 1
    clusters_w = clusters[:,2] - clusters[:,0] + 1
    normal_idx = np.where(clusters[:,4] == 1)
    global os
    global glob
    import os
    # pre In[7]
    import glob
imports()
import sys
import glob
from statistics import mode
from numpy import linalg as la
import cv2
from keras.models import load_model
import numpy as np
import pickle
import skvideo.io
from keras.preprocessing import image
from utils.datasets import get_labels
from utils.inference import load_image
from utils.preprocessor import preprocess_input

font = cv2.FONT_HERSHEY_SIMPLEX
# hyper-parameters for bounding boxes shape
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces as face_detect
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input



#dir_to_save='./crops/'
# detect_faces('1510871658-16649592-5447239cde1e8e618e21fd9a42e51e884dd25134-6d3d7c42.jpg', dir_to_save)
def detect_face_emotions_and_draw(frame):
    # parameters for loading data and images
    detection_model_path = './trained_models/detection_models/haarcascade_frontalface_default.xml'
    emotion_model_path = './trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    emotion_labels = get_labels('fer2013')

    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (15, 35)

    # loading models
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # starting lists for calculating modes
    emotion_window = []

    bgr_image = frame
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('temp.bmp', frame)
    rect_image, faces = detect_faces('temp.bmp','./crops/')
    cv2.imwrite('temp_rect.bmp', rect_image)
    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        print(x1, x2, y1, y2, emotion_target_size)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            print('exception ignored while resize')
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        #try:
            #emotion_mode = mode(emotion_window)
        #except:
           # print('exception ignored for emotion mode')
           # continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()
        print(emotion_text)
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_text,
                  color, 0, 50, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return rgb_image

def load_model_emb():
    emotion_model_path = 'fer2013_mini_XCEPTION.102-0.66.hdf5'
    emotion_labels = get_labels('fer2013')
    font = cv2.FONT_HERSHEY_SIMPLEX

    # hyper-parameters for bounding boxes shape
    emotion_offsets = (20, 40)
    emotion_offsets = (0, 0)

    # loading modelsf
    emotion_classifier = load_model(emotion_model_path)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]
    emotion_classifier.layers.pop()
    emotion_classifier.layers.pop()
    emotion_classifier.outputs = [emotion_classifier.layers[-1].output]
    emotion_classifier.layers[-1].outbound_nodes = []
    emotion_classifier.summary()
    return emotion_classifier, emotion_target_size

# emotion_classifier,emotion_target_size = load_model_emb()
fully = False

def detect_faces(img_path, dir_to_save):
    print ('started with ',str(img_path))
    raw_img = cv2.imread(img_path)
    raw_h = raw_img.shape[0]
    raw_w = raw_img.shape[1]
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    raw_img_f = raw_img.astype(np.float32)
    
    min_scale = min(np.floor(np.log2(np.max(clusters_w[normal_idx]/raw_w))), np.floor(np.log2(np.max(clusters_h[normal_idx]/raw_h))))
    max_scale = min(1.0, -np.log2(max(raw_h, raw_w)/MAX_INPUT_DIM))


    # In[9]:

    scales_down = pl.frange(min_scale, 0, 1.)
    scales_up = pl.frange(0.5, max_scale,0.5)
    scales_pow = np.hstack((scales_down, scales_up))
    scales = np.power(2.0, scales_pow)


    # In[10]:

    context=mx.cpu()


    # In[12]:

    mod = mx.mod.Module(symbol=all_layers['fusex_output'], context=context, data_names=['data'], label_names=None)
    mod.bind(for_training=False,
                data_shapes=[('data', (1, 3, 224, 224))],
                label_shapes=None,
                force_rebind=False)
    mod.set_params(arg_params=arg_params, aux_params=aux_params, force_init=False)


    # In[13]:


    start = time.time()
    bboxes = np.empty(shape=(0,5))
    print (scales.size,' scales')
    for s in scales:
        print ('scale ',s,' in process')
        img = cv2.resize(raw_img_f, (0,0), fx = s, fy = s)
        img = np.transpose(img,(2,0,1))
        img = img - averageImage

        tids = []
        if s <= 1. :
            tids = list(range(4, 12))
        else :
            tids = list(range(4, 12)) + list(range(18, 25))
        ignoredTids = list(set(range(0,clusters.shape[0]))-set(tids))
        img_h = img.shape[1]
        img_w = img.shape[2]
        img = img[np.newaxis, :]

        mod.reshape(data_shapes=[('data', (1, 3, img_h, img_w))])
        mod.forward(Batch([mx.nd.array(img)]))
        mod.get_outputs()[0].wait_to_read()
        fusex_res = mod.get_outputs()[0]

        score_cls = mx.nd.slice_axis(fusex_res, axis=1, begin=0, end=25, name='score_cls')
        score_reg = mx.nd.slice_axis(fusex_res, axis=1, begin=25, end=None, name='score_reg')
        prob_cls = mx.nd.sigmoid(score_cls)

        prob_cls_np = prob_cls.asnumpy()
        prob_cls_np[0,ignoredTids,:,:] = 0.

        _, fc, fy, fx = np.where(prob_cls_np > prob_thresh)

        cy = fy * 8 - 1
        cx = fx * 8 - 1
        ch = clusters[fc, 3] - clusters[fc,1] + 1
        cw = clusters[fc, 2] - clusters[fc, 0] + 1

        Nt = clusters.shape[0]

        score_reg_np = score_reg.asnumpy()
        tx = score_reg_np[0, 0:Nt, :, :]
        ty = score_reg_np[0, Nt:2*Nt,:,:]
        tw = score_reg_np[0, 2*Nt:3*Nt,:,:]
        th = score_reg_np[0,3*Nt:4*Nt,:,:]

        dcx = cw * tx[fc, fy, fx]
        dcy = ch * ty[fc, fy, fx]
        rcx = cx + dcx
        rcy = cy + dcy
        rcw = cw * np.exp(tw[fc, fy, fx])
        rch = ch * np.exp(th[fc, fy, fx])

        score_cls_np = score_cls.asnumpy()
        scores = score_cls_np[0, fc, fy, fx]

        tmp_bboxes = np.vstack((rcx-rcw/2, rcy-rch/2, rcx+rcw/2,rcy+rch/2))
        tmp_bboxes = np.vstack((tmp_bboxes/s, scores))
        tmp_bboxes = tmp_bboxes.transpose()
        bboxes = np.vstack((bboxes, tmp_bboxes))
                # if((time.time()-start)>60.0):
                #     return


            # In[14]:

    print ("time", time.time()-start, "secs.")
    refind_idx = nms(bboxes, nms_thresh)
    refind_bboxes = bboxes[refind_idx]
    refind_bboxes = refind_bboxes.astype(np.int32)
            # fig = plt.figure()
    original_image = cv2.imread(img_path)
    i=0
    faces=[]
    faces_coordinates=[]
    for r in refind_bboxes:
                # print 'crop face file name is ',str(p[2:]).replace('/','')+'face'+str(i)+'.bmp'
#         new_name=dir_to_save+str(p[2:]).replace('/','')+'face'+str(i)+'.bmp'
        new_name=dir_to_save+os.path.basename(img_path)+'face'+str(i)+'.bmp'
        crop_image=original_image[r[1]:r[3],r[0]:r[2]]
        faces.append(crop_image)
        faces_coordinates.append([r[0], r[1], r[2]-r[0], r[3]-r[1]])
        cv2.imwrite(new_name,crop_image)
        print('saved to '+dir_to_save+os.path.basename(img_path)+'face'+str(i)+'.bmp')
        i=i+1
        cv2.rectangle(original_image, (r[0],r[1]), (r[2],r[3]), (255,255,0),3)
    return original_image, faces_coordinates
            #     cv2.rectangle(raw_img, (r[0],r[1]), (r[2],r[3]), (255,255,0),3)
            # plt.imshow(raw_img)
            # plt.show()
#             os.rename(p, str(p) + 'd')

            # In[15]:
#             print j,' out of ',len(pictures_list), 'done'
def euqMean(array, axis):
    norm=la.norm(array, axis=0)
    result=[]
    for num in array:
        result.append(num/norm)
    return np.mean(result, axis)

def meanEuq(array, axis):
    pre=np.mean(array, axis=0)
    norm=la.norm(pre, axis=0)
    result=[]
    for num in pre:
        result.append(num/norm)
    return result

rf=pickle.load(open('RandomForestClassifier_best_euqMean_emb_0.708645677161', 'rb'))
grb=pickle.load(open('GradientBoostingClassifier_best_euqMean_emb_0.700649675162', 'rb'))
ada=pickle.load(open('AdaBoostClassifier_best_euqMean_emb_0.709145427286', 'rb'))
bag=pickle.load(open('BaggingClassifier_best_euqMean_emb_0.706646676662', 'rb'))
xtr=pickle.load(open('ExtraTreesClassifier_best_euqMean_emb_0.702148925537', 'rb'))
lsvc=pickle.load(open('LinearSVC_best_mean_emb_0.702648675662', 'rb'))
svc=pickle.load(open('SVC_best_mean_emb_0.692653673163', 'rb'))
clf_dict={
            #rf:[90, euqMean,'emb']
#           ,grb:[112, 'euqMean','emb']
          #,svc:[80,np.mean,'emb']
          #,lsvc:[91,meanEuq,'emb']
#           ,ada:[90,'euqMean','emb']
#           ,bag:[97,'euqMean','emb']
#           ,xtr:[51,'euqMean','emb']
         }

emotion_classifier,emotion_target_size = load_model_emb()

def vote_decision(answers):
    positive_count=answers.count(0)
    neutral_count=answers.count(1)
    negative_count=answers.count(2)
#     print(positive_count)
#     print(neutral_count)
#     print(negative_count)
    if positive_count>=neutral_count and positive_count>=negative_count:
        return 'positive'
    elif neutral_count>=positive_count and neutral_count>=negative_count:
        return 'neutral'
    elif negative_count>=positive_count and negative_count>=neutral_count:
        return 'negative'

def mean_decision(answers):
    inv_answers={0:'positive', 1:'neutral', 2:'negative'}
    final_answer=inv_answers[round(np.mean(answers))]
    return final_answer

def median_decision(answers):
    inv_answers={0:'positive', 1:'neutral', 2:'negative'}
    final_answer=inv_answers[round(np.median(answers))]
    return final_answer

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.winfo_toplevel().title("final_solution")
        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Choose file"
        self.hi_there["command"] = self.choose_pic
        self.hi_there.pack(side="left")
        self.quit = tk.Button(self, text="Quit",
                              command=root.destroy)
        self.quit.pack(side="right")

        self.recognize_b = tk.Button(self)
        self.recognize_b["text"] = "Recognize"
        self.recognize_b["command"] = self.recognize
        self.recognize_b.pack(side='right')

        cb_frame=Frame(self)

        self.hi_there.pack(side="left")

        self.myvar = Label(root)
        self.emotion_label = Label(root, text='Choose file for group emotion recognition')
        self.emotion_label.pack(side='bottom')
        self.path = ''

        self.rfVar = IntVar()
        self.grbVar = IntVar()
        self.adaVar = IntVar()
        self.bagVar = IntVar()
        self.xtrVar = IntVar()
        self.lsvcVar = IntVar()
        self.svcVar = IntVar()

        rf_cb=Checkbutton(cb_frame, text = "RandomForestClassifier", variable = self.rfVar, onvalue = 1, offvalue = 0)
        grb_cb=Checkbutton(cb_frame, text = "GradientBoostingClassifier", variable = self.grbVar, onvalue = 1, offvalue = 0)
        ada_cb=Checkbutton(cb_frame, text = "AdaBoostClassifier", variable = self.adaVar, onvalue = 1, offvalue = 0)
        bag_cb=Checkbutton(cb_frame, text = "BaggingClassifier", variable = self.bagVar, onvalue = 1, offvalue = 0)
        xtr_cb=Checkbutton(cb_frame, text = "ExtraTreesClassifier", variable = self.xtrVar, onvalue = 1, offvalue = 0)
        lsvc_cb=Checkbutton(cb_frame, text = "LinearSVC", variable = self.lsvcVar, onvalue = 1, offvalue = 0)
        svc_cb=Checkbutton(cb_frame, text = "SVC", variable = self.svcVar, onvalue = 1, offvalue = 0)

        rf_cb.grid(row=0, sticky='w')
        grb_cb.grid(row=1, sticky='w')
        ada_cb.grid(row=2, sticky='w')
        bag_cb.grid(row=3, sticky='w')
        xtr_cb.grid(row=4, sticky='w')
        lsvc_cb.grid(row=5, sticky='w')
        svc_cb.grid(row=6, sticky='w')

        rb_frame=Frame(self)
        rb_frame.pack(side='right')

        cb_frame.pack(side='right')

        self.decision_var = object()

        def sel(dec):
           self.decision_make=dec

        mean_rb = Radiobutton(rb_frame, text="Mean decision", variable=self.decision_var, value=mean_decision,
                          command=sel(mean_decision)
)
        mean_rb.grid(row=0, sticky='w')

        median_rb = Radiobutton(rb_frame, text="Median decision", variable=self.decision_var, value=median_decision,
                          command=sel(median_decision)
)
        median_rb.grid(row=1, sticky='w')

        vote_rb = Radiobutton(rb_frame, text="Vote decision", variable=self.decision_var, value=vote_decision,
                          command=sel(vote_decision)
)
        vote_rb.grid(row=2, sticky='w')



    def choose_pic(self):
        
        self.path=filedialog.askopenfilename(filetypes=[("Image File",'.*')])  
        im = []
        temp_img_name='temp_to_show.bmp'
        if self.path.endswith('.mp4'):
            vid = skvideo.io.vreader(self.path)
            count=0
            for frame in vid:
              cv2.imwrite("./crops/frame%d.bmp" % count, frame)     # save frame as JPEG file      
              count += 1
              im=frame
            cv2.imwrite(temp_img_name, detect_face_emotions_and_draw(im))
            im = Image.open(temp_img_name)
        else:
            im = Image.open(self.path)
        tkimage = ImageTk.PhotoImage(im)
        self.myvar.config(image = tkimage)
        self.myvar.image = tkimage
        self.myvar.pack()
        self.emotion_label.config(text='Press rezognize for group emotion recognition', background='White')

        
    def recognize_old(self):
            #rf:[90, euqMean,'emb']
#           ,grb:[112, 'euqMean','emb']
          #,svc:[80,np.mean,'emb']
          #,lsvc:[91,meanEuq,'emb']
#           ,ada:[90,'euqMean','emb']
#           ,bag:[97,'euqMean','emb']
#           ,xtr:[51,'euqMean','emb']
        if self.rfVar.get():
            clf_dict[rf]=[90, euqMean,'emb']
        if self.grbVar.get():
            clf_dict[grb]=[112, euqMean,'emb']
        if self.adaVar.get():
            clf_dict[ada]=[90,euqMean,'emb']
        if self.bagVar.get():
            clf_dict[bag]=[97,euqMean,'emb']
        if self.xtrVar.get():
            clf_dict[xtr]=[51, euqMean,'emb']
        if self.lsvcVar.get():
            clf_dict[lsvc]=[91,meanEuq,'emb']
        if self.svcVar.get():
            clf_dict[svc]=[80,np.mean,'emb']

        dir_to_save='./crops/'

        #new_photo=detect_faces(self.path, dir_to_save)
        #cv2.imwrite(dir_to_save+os.path.basename(self.path)+'new.bmp', new_photo)
        faces=glob.glob(dir_to_save+'*'+os.path.basename(self.path)+'*')
        for image_path in faces:
            print('taking '+image_path+' into account')
            try:
                gray_image = load_image(image_path, grayscale=True)
            except:
                continue
            gray_image = np.squeeze(gray_image)
            gray_image = gray_image.astype('uint8')

            gray_face = gray_image

            gray_face = cv2.resize(gray_face, (emotion_target_size))

            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                print('exception ignored')

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, axis=0)
            gray_face = np.expand_dims(gray_face, -1)
        #     if 'fer' in layer_label:
        #         gray_face = np.expand_dims(gray_face, 0)
        #     if 'fer' not in layer_label:
        #         gray_face = np.expand_dims(gray_face, -1)
            dict_faces[str(image_path)]=emotion_classifier.predict(gray_face)
        # print(dict_faces)
        answers={'positive':0, 'neutral':1, 'negative':2}
        ans_arr=[]
        for clf in clf_dict:
            pca=pickle.load(open('best_pca_'+clf.__class__.__name__+'_n'+str(clf_dict[clf][0]),'rb'))
            vectors=[]
            for key in dict_faces.keys():
                vectors.append(dict_faces[key])
            vectors=np.reshape(vectors,(len(vectors),-1))
            print(clf.__class__.__name__)
            print(clf_dict[clf][1].__class__.__name__)
            feature_vector=clf_dict[clf][1](vectors, axis=0)
        #     print('pca_after\n')
        #     print(pca.transform(np.expand_dims(feature_vector, axis=0))[0])
        #     print('pca_before\n')
        #     print(feature_vector)
            ans_arr.append(answers[clf.predict(pca.transform(np.expand_dims(feature_vector, axis=0)))[0]])
        self.emotion_label.config(text=('Emotion: '+self.decision_make(ans_arr)).upper(), font='Times 30')
        im = Image.open(dir_to_save+os.path.basename(self.path)+'new.bmp')
        tkimage = ImageTk.PhotoImage(im)
        self.myvar.config(image = tkimage)
        self.myvar.image = tkimage
        self.myvar.pack()
        
    def recognize(self):
            #rf:[90, euqMean,'emb']
#           ,grb:[112, 'euqMean','emb']
          #,svc:[80,np.mean,'emb']
          #,lsvc:[91,meanEuq,'emb']
#           ,ada:[90,'euqMean','emb']
#           ,bag:[97,'euqMean','emb']
#           ,xtr:[51,'euqMean','emb']
        #self.path='1output.mp4'
        if self.rfVar.get():
            clf_dict[rf]=[90, euqMean,'emb']
        if self.grbVar.get():
            clf_dict[grb]=[112, euqMean,'emb']
        if self.adaVar.get():
            clf_dict[ada]=[90,euqMean,'emb']
        if self.bagVar.get():
            clf_dict[bag]=[97,euqMean,'emb']
        if self.xtrVar.get():
            clf_dict[xtr]=[51, euqMean,'emb']
        if self.lsvcVar.get():
            clf_dict[lsvc]=[91,meanEuq,'emb']
        if self.svcVar.get():
            clf_dict[svc]=[80,np.mean,'emb']
        inv_answers={0:'positive', 1:'neutral', 2:'negative'}
        dir_to_save='./crops/'
        if self.path.endswith('.mp4'):
            vid = skvideo.io.vread(self.path)
            frames_answers=[]
            frame_count=0
            for i in range(0,vid.shape[0]):
                frame = cv2.cvtColor(vid[i],cv2.COLOR_BGR2RGB)
                dict_faces={}
                frame_count+=1
                print(str(frame_count)+' frame')
                face_count=0
                for face in mtcnn.detect_faces(frame):
                    try:
                        gray_image=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                        gray_image=image.img_to_array(gray_image)
                    except:
                        continue
                    gray_image = np.squeeze(gray_image)
                    gray_image = gray_image.astype('uint8')

                    gray_face = gray_image

                    gray_face = cv2.resize(gray_face, (emotion_target_size))

                    try:
                        gray_face = cv2.resize(gray_face, (emotion_target_size))
                    except:
                        print('exception ignored')

                    gray_face = preprocess_input(gray_face, True)
                    gray_face = np.expand_dims(gray_face, axis=0)
                    gray_face = np.expand_dims(gray_face, -1)
                #     if 'fer' in layer_label:
                #         gray_face = np.expand_dims(gray_face, 0)
                #     if 'fer' not in layer_label:
                #         gray_face = np.expand_dims(gray_face, -1)
                    dict_faces['face'+str(face_count)]=emotion_classifier.predict(gray_face)
                    face_count+=1
                #print(face_count)
                answers={'positive':0, 'neutral':1, 'negative':2}
                ans_arr=[]
                #print(dict_faces)
                for clf in clf_dict:
                    pca=pickle.load(open('best_pca_'+clf.__class__.__name__+'_n'+str(clf_dict[clf][0]),'rb'))
                    vectors=[]
                    for key in dict_faces.keys():
                        vectors.append(dict_faces[key])
                    vectors=np.reshape(vectors,(len(vectors),-1))
                    #print(clf.__class__.__name__)
                    #print(clf_dict[clf][1].__class__.__name__)
                    feature_vector=clf_dict[clf][1](vectors, axis=0)
                #     print('pca_after\n')
                #     print(pca.transform(np.expand_dims(feature_vector, axis=0))[0])
                #     print('pca_before\n')
                #     print(feature_vector)
                    ans_arr.append(answers[clf.predict(pca.transform(np.expand_dims(feature_vector, axis=0)))[0]])
                frames_answers.append(answers[self.decision_make(ans_arr)])
            final_decison=inv_answers[max(set(frames_answers), key=frames_answers.count)]
            print(final_decison)
        self.emotion_label.config(text=('Group Emotion: '+final_decison).upper(), font='Times 30')
        #im = Image.open(dir_to_save+os.path.basename(self.path)+'new.bmp')
        #tkimage = ImageTk.PhotoImage(im)
        #self.myvar.config(image = tkimage)
        #self.myvar.image = tkimage
        self.myvar.pack()

root = tk.Tk()
root.style=ttk.Style()
root.style.theme_use('clam')
app = Application(master=root)
app.mainloop()
#app.recognize()
