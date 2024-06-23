import numpy as np
import string
import cv2
from tkinter import *
from tkinter import filedialog
import PIL.Image
import PIL.ImageTk
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from pickle import load
import pyttsx3
from googletrans import Translator

# Load the models and tokenizer
cnn_model = load_model("CNN_Model.h5")
rnn_model = load_model("RNN_Model.h5")
tokenizer = load(open("Flickr8K_Tokenizer.p", "rb"))
word_to_index = tokenizer.word_index
index_to_word = dict([index, word] for word, index in word_to_index.items())
vocab_size = len(tokenizer.word_index) + 1
max_len = 31

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize Translator
translator = Translator()

# Function to speak the caption
def speakCaption(caption):
    engine.say(caption)
    engine.runAndWait()

# Function to translate caption
def translateCaption(caption, target_language):
    translated_caption = translator.translate(caption, dest=target_language).text
    return translated_caption

# Initialize Tkinter
root = Tk()
root.title("Image Caption Generator")
root.state('zoomed')
root.resizable(width=True, height=True)

# Label for the application title
panel = Label(root, text='IMAGE CAPTION GENERATOR', font=("Arial", 30))
panel.place(relx=0.3, rely=0.1)

filename = None

# Function to choose an image
def chooseImage(event=None):
    global filename
    filename = filedialog.askopenfilename()
    img = PIL.Image.open(filename)
    img = img.resize((350, 300))
    img = PIL.ImageTk.PhotoImage(img)
    display_image = Label(root, image=img)
    display_image.image = img
    display_image.place(relx=0.4, rely=0.2)

# Function to capture image from webcam
def captureImage(event=None):
    global filename
    camera = cv2.VideoCapture(0)
    return_value, image = camera.read()
    cv2.imwrite("captured_image.png", image)
    camera.release()
    filename = "captured_image.png"
    img = PIL.Image.open(filename)
    img = img.resize((350, 300))
    img = PIL.ImageTk.PhotoImage(img)
    display_image = Label(root, image=img)
    display_image.image = img
    display_image.place(relx=0.4, rely=0.2)

value = StringVar()
target_language = StringVar()

# Function to generate caption
def generateCaption(event=None):
    if(filename == None):
        value.set("No Image Selected")
    else:
        img = load_img(filename, target_size=(299, 299))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 127.5
        img = img - 1.0
        features = cnn_model.predict(img)
        in_text = 'startseq'
        for i in range(max_len):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=31)
            pred = rnn_model.predict([features, sequence], verbose=0)
            pred = np.argmax(pred)
            word = index_to_word[pred]
            if word is None:
                break
            in_text += ' ' + word
            if word == 'endseq':
                break
        in_text = ' '.join(in_text.split()[1: -1])
        in_text = in_text[0].upper() + in_text[1:] + '.'
        value.set(in_text)
    display_caption = Label(root, textvariable=value, font=("Arial", 18))
    display_caption.place(relx=0.48, rely=0.85)

# Function to speak caption
def speakCaptionWrapper(event=None):
    caption = value.get()
    if caption:
        speakCaption(caption)

# Function to translate caption
def translateCaptionWrapper(event=None):
    caption = value.get()
    if caption:
        translated_caption = translateCaption(caption, target_language.get())
        value.set(translated_caption)

# Button to choose image
button1 = Button(root, text='Choose an Image', font=(None, 18), activeforeground='red', bd=10, relief=RAISED, height=2, width=15, command=chooseImage)
button1.place(relx=0.3, rely=0.65)

# Button to capture image
button2 = Button(root, text='Capture Image', font=(None, 18), activeforeground='red', bd=10, relief=RAISED, height=2, width=15, command=captureImage)
button2.place(relx=0.5, rely=0.65)

# Button to generate caption
button3 = Button(root, text='Generate Caption', font=(None, 18), activeforeground='red', bd=10, relief=RAISED, height=2, width=15, command=generateCaption)
button3.place(relx=0.7, rely=0.65)

# Button to speak caption
button4 = Button(root, text='Speak Caption', font=(None, 18), activeforeground='red', bd=10, relief=RAISED, height=2, width=15, command=speakCaptionWrapper)
button4.place(relx=0.83, rely=0.90)

# Option menu for selecting target language
languages = ["en", "fr", "es", "de", "hi", "bn", "it", "pt", "ru"]  # English, French, Spanish, German, Hindi, Bengali, Italian, Portuguese, Russian
option_menu = OptionMenu(root, target_language, *languages)
option_menu.config(font=("Arial", 14))
option_menu.place(relx=0.3, rely=0.90)

# Button to translate caption
button5 = Button(root, text='Translate Caption', font=(None, 18), activeforeground='red', bd=10, relief=RAISED, height=2, width=15, command=translateCaptionWrapper)
button5.place(relx=0.5, rely=0.90)

caption = Label(root, text='Caption : ', font=("Arial", 18))
caption.place(relx=0.35, rely=0.85)

root.mainloop()
