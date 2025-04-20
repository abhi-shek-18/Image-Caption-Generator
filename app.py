from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from deep_translator import GoogleTranslator

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'static/audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        lang = request.form.get('lang', 'en')

        # Save image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load and process image
        raw_image = Image.open(filepath).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt")

        # Generate caption
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        # Translate caption
        translated_caption = GoogleTranslator(source='en', target=lang).translate(caption)

        # Generate TTS for translated caption
        audio_filename = f"{os.path.splitext(file.filename)[0]}_{lang}.mp3"
        audio_path = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
        tts = gTTS(text=translated_caption, lang=lang)
        tts.save(audio_path)

        return jsonify({
            'caption': caption,
            'translated_caption': translated_caption,
            'image_url': f"/uploads/{file.filename}",
            'audio_url': f"/static/audio/{audio_filename}"
        })

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("Starting the Flask app...")
    app.run(debug=True)
