# from tkinter import *
from tkinter import Button, Label, Frame, Tk,Entry,END
import PIL
from PIL import Image, ImageGrab, ImageTk
import numpy as np
from tkinter import filedialog
import os.path
from os import path
import xlwt
from tkinter import messagebox
import cv2
import os.path
import glob
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
import imutils
import xlwt
import pandas as pd
import xlsxwriter
from tkinter.ttk import Progressbar
import time
import pyperclip
from scipy.spatial.distance import cosine

tf.compat.v1.disable_eager_execution()
from mtcnn.mtcnn import MTCNN
import random

img_1 = np.uint8(np.zeros((480, 640, 3)))
img_2 = np.uint8(np.zeros((480, 640, 3)))

window = Tk()
window.title("People Searcher")
window.geometry('1300x900')

blank = np.uint8(np.zeros((480, 640, 3)))
imgtk = ImageTk.PhotoImage(image=PIL.Image.fromarray(blank))
tk_blank = ImageTk.PhotoImage(ImageTk.Image.fromarray(blank))

blank = np.uint8(np.zeros((480, 640, 3)))
imgtk2 = ImageTk.PhotoImage(image=PIL.Image.fromarray(blank))
tk_blank2 = ImageTk.PhotoImage(ImageTk.Image.fromarray(blank))

detector = MTCNN()
face_net = load_model("facenet_keras.h5")
face_net.compile(
    optimizer='adam',
    loss="binary_crossentropy",
    metrics=['accuracy']
)


def extract_faces(img, target_size):
    H = np.shape(img)[0]
    W = np.shape(img)[1]
    blank = np.zeros((112, 112, 3))
    results = detector.detect_faces(img)
    faces = []
    bounds = []
    if len(results) == 0:
        return False, [], []
    for i in range(len(results)):
        x1, y1, width, height = results[i]['box']
        x2, y2 = x1 + width, y1 + height
        if y1 < 0:
            y1 = 0
            y2 = y2 + y1
        if x1 < 0:
            x1 = 0
            x2 = x2 + x1
        # extract the face
        face = img[y1:y2, x1:x2]
        # resize pixels to the model size
        image2 = cv2.resize(face, target_size)
        face_array = np.array(image2)
        faces += [face_array]
        # normalize bounds
        xmin = x1 / W
        ymin = y1 / H
        xmax = x2 / W
        ymax = y2 / H
        bound = [xmin, ymin, xmax, ymax]
        bounds += [bound]
    return True, faces, bounds


# prepares data to run on facenet
def prepare_img(img):
    img = cv2.resize(img, (160, 160))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_pixels = img.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # face_pixels=np.expand_dims(face_pixels,axis=0)
    return face_pixels


def extract_face_embeddings(img_array):
    global face_net
    prepared_images = []
    for i in range(len(img_array)):
        prepared_images += [prepare_img(img_array[i])]
    prepared_images = np.array(prepared_images, dtype=np.float32)
    tf.function(experimental_relax_shapes=True)
    embeddings = face_net.predict(prepared_images)
    return embeddings


def extract_face_embeddings_image(img):
    success, faces, bounds = extract_faces(img, (160, 160))
    if not success:
        return False, np.zeros((128)), np.zeros((4))
    embeddings = extract_face_embeddings(faces)
    return True, embeddings, bounds


def readjust_bounds(img, bound):
    height = np.shape(img)[0]
    width = np.shape(img)[1]
    xmin = int(bound[0] * width)
    ymin = int(bound[1] * height)
    xmax = int(bound[2] * width)
    ymax = int(bound[3] * height)
    return xmin, ymin, xmax, ymax


# draw boxes on the drawing
def draw_boxes(img, bounds, color_codes):
    # seperate the image as a seperate object
    img_array = np.uint8(np.array(img))
    for i in range(len(bounds)):
        x1, y1, x2, y2 = readjust_bounds(img_array, bounds[i])
        color = color_codes[i]
        img_array = cv2.rectangle(img_array, (x1, y1), (x2, y2), color_codes[i], 2)
    return img_array


def compare_embeddings(embeddings_1, embeddings_2, bounds1, bounds2, color_codes, threshold=.5):
    c_codes_1 = []
    b_bounds_1 = []

    c_codes_2 = []
    b_bounds_2 = []

    prox_test = []

    a = embeddings_1
    b = embeddings_2
    success = False
    for i in range(len(embeddings_1)):
        for j in range(len(embeddings_2)):
            proximity = 1 - cosine(embeddings_1[i], embeddings_2[j])
            prox_test += [proximity]
            if proximity >= threshold:
                success = True
                c_codes_2 += [color_codes[i]]
                b_bounds_2 += [bounds2[j]]
                c_codes_1 += [color_codes[i]]
                b_bounds_1 += [bounds1[i]]

    return success, b_bounds_1, c_codes_1, b_bounds_2, c_codes_2,


def compare_for_faces():
    global img_1, img_2
    A, emb_1, b1 = extract_face_embeddings_image(img_1)
    B, emb_2, b2 = extract_face_embeddings_image(img_2)
    if (not A):
        print("no face detected for image 1")
        return img_1, img_2
    if (not B):
        print("no face detected for image 2")
        return img_1, img_2

    color_codes = []
    pre_codes = [
        (255, 0, 0),
        (0, 254, 0),
        (0, 0, 254),
        (255, 255, 0),
        (255, 0, 255),
        (0, 0, 255),
        (0, 0, 0),
        (255, 255, 255),
        (192, 192, 192),
        (128, 128, 128),
        (128, 0, 0),
        (128, 128, 0),
        (0, 128, 0),
        (128, 0, 128),
        (0, 128, 128),
        (0, 0, 128)
    ]
    for c in range(len(emb_1)):
        # assign color codes according to a predetermin order
        if c < len(pre_codes):
            color_codes += [pre_codes[c]]
        else:
            # if number of faces exceeds the pre codes start making
            # random colors
            b = random.randint(0, 255)
            g = random.randint(0, 255)
            r = random.randint(0, 255)
            color_codes += [(b, g, r)]
    # def compare_embeddings(embeddings_1, embeddings_2, bounds1, bounds2, color_codes, threshold=.3):
    success, b_bounds_1, c_codes_1, b_bounds_2, c_codes_2 = compare_embeddings(emb_1, emb_2, b1, b2, color_codes)
    if not success:
        return img_1, img_2

    box1 = draw_boxes(img_1, b_bounds_1, c_codes_1)
    box2 = draw_boxes(img_2, b_bounds_2, c_codes_2)

    imgtk_1 = ImageTk.PhotoImage(image=PIL.Image.fromarray(box1))
    left_pic_lbl.configure(image=imgtk_1)
    left_pic_lbl.image = imgtk_1

    imgtk_2 = ImageTk.PhotoImage(image=PIL.Image.fromarray(box2))
    right_pic_lbl.configure(image=imgtk_2)
    right_pic_lbl.image = imgtk_2


def load_left():
    global left_pic, img_1
    source = filedialog.askopenfilename(initialdir="C:/Users/Bruno/PycharmProjects/BDproject1/face finder/pics/", title="Select file",
                                        filetypes=(
                                            ("jpg files", "*.jpg"), ("png files", "*.png"), ("all files", "*.*")))
    if source == '':
        return
    img = cv2.imread(source)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_1 = cv2.resize(img, (640, 480))
    update_images()


def load_right():
    global right_pic, img_2
    source = filedialog.askopenfilename(initialdir="C:/Users/Bruno/PycharmProjects/BDproject1/face finder/pics/", title="Select file",
                                        filetypes=(
                                             ("jpg files", "*.jpg"), ("png files", "*.png"),("all files", "*.*")))
    if source == '':
        return
    img = cv2.imread(source)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_2 = cv2.resize(img, (640, 480))
    update_images()


def update_images():
    global right_pic_lbl, left_pic_lbl, img_1, img_2
    test = left_pic_lbl
    imgtk_1 = ImageTk.PhotoImage(image=PIL.Image.fromarray(img_1))
    left_pic_lbl.configure(image=imgtk_1)
    left_pic_lbl.image = imgtk_1

    imgtk_2 = ImageTk.PhotoImage(image=PIL.Image.fromarray(img_2))
    right_pic_lbl.configure(image=imgtk_2)
    right_pic_lbl.image = imgtk_2


left_frame = Frame(master=window)
left_frame.grid(column=0, row=0)

center_frame = Frame(master=window)
center_frame.grid(column=1, row=0)

right_frame = Frame(master=window)
right_frame.grid(column=2, row=0)

load_btn_left = Button(master=left_frame, text="load image", command=load_left)
load_btn_left.grid(column=0, row=0)

star_btn = Button(master=center_frame, text="start", command=compare_for_faces)
star_btn.grid(column=0, row=0)

threshold_lbl=Label(master=center_frame, text="threshold")
threshold_lbl.grid(column=0, row=1)

threshold_txt = Entry(master=center_frame)
threshold_txt.grid(column=0, row=2)
threshold_txt.insert(END, ".5")

load_btn_right = Button(master=right_frame, text="load image", command=load_right)
load_btn_right.grid(column=0, row=0)

left_pic_lbl = Label(master=left_frame, image=tk_blank)
left_pic_lbl.grid(column=0, row=1, padx=5, pady=5)

right_pic_lbl = Label(master=right_frame, image=tk_blank2)
right_pic_lbl.grid(column=0, row=1, padx=5, pady=5)

window.mainloop()
