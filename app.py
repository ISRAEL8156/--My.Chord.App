# -*- coding: utf-8 -*-
import os
import json
import time
import threading
import webbrowser
import hashlib
import traceback
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import librosa
import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile as sf
from pedalboard import Pedalboard, PitchShift
import urllib.parse

try:
    from mido import Message, MidiFile, MidiTrack
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# נעילת נתיבים לתיקיית הפרויקט
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
LIB_DIR = os.path.join(BASE_DIR, "chord_library")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LIB_DIR, exist_ok=True)

NOTES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

def get_note_name(idx, prefer_flats=False):
    if prefer_flats: return NOTES_FLAT[idx % 12]
    return NOTES_SHARP[idx % 12]

def get_chord_keys(chord):
    if not chord: return []
    suffix = ""
    root_str = chord
    for s in ["m7", "maj7", "7", "m", "dim"]:
        if chord.endswith(s):
            suffix = s
            root_str = chord.replace(s, "")
            break
            
    if "b" in root_str: root_list = NOTES_FLAT
    else: root_list = NOTES_SHARP
        
    if root_str not in root_list: return []
    r_idx = root_list.index(root_str)
    intervals_map = {"": [0, 4, 7], "m": [0, 3, 7], "7": [0, 4, 7, 10], "m7": [0, 3, 7, 10], "maj7": [0, 4, 7, 11], "dim": [0, 3, 6]}
    intervals = intervals_map.get(suffix, [0, 4, 7])
    return [(r_idx + i) % 12 for i in intervals]

def get_file_hash(filepath, settings_str=""):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    hasher.update(settings_str.encode('utf-8'))
    return hasher.hexdigest()

def analyze_audio_file(file_path, hop=512, stay_prob=0.96, min_dur=0.4, complex_penalty=0.6):
    y, sr = librosa.load(file_path, sr=11025)
    
    # המנוע המקורי והטוב! ללא סינכרון פעימות או סינון מוגזם
    b, a = scipy.signal.butter(4, 100 / (sr / 2), btype='high')
    y_filtered = scipy.signal.filtfilt(b, a, y)
    y_harm = librosa.effects.harmonic(y_filtered, margin=3.0)
    tuning = librosa.estimate_tuning(y=y_harm, sr=sr)
    
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, tuning=tuning, hop_length=hop)
    chroma = scipy.ndimage.median_filter(chroma, size=(1, 11))
    
    templates = []
    types = [("", [1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0]), ("m", [1.0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0]), 
             ("7", [1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0.8, 0]), ("dim", [1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0, 0])]
    for t_name, t_pattern in types:
        for i in range(12): templates.append((t_name, np.roll(t_pattern, i)))
            
    probs = np.dot([t[1] for t in templates], chroma)
    
    # החלת העונש על אקורדים מורכבים לפי בחירת המשתמש
    probs[24:36, :] *= complex_penalty 
    probs[36:48, :] *= (complex_penalty * 0.7) 
    probs /= (probs.max(axis=0) + 1e-6)
    
    transition_matrix = np.eye(len(templates)) * stay_prob + (1 - stay_prob) / len(templates)
    best_path = librosa.sequence.viterbi(probs, transition_matrix)
    
    results = []
    if len(best_path) > 0:
        last_idx = best_path[0]
        start_t = 0.0
        for t, idx in enumerate(best_path):
            if idx != last_idx:
                curr_t = librosa.frames_to_time(t, sr=sr, hop_length=hop)
                if curr_t - start_t > min_dur:
                    chord_name = get_note_name(last_idx % 12) + templates[last_idx][0]
                    results.append({"chord": chord_name, "start": float(start_t), "duration": float(curr_t - start_t)})
                    start_t = curr_t
                    last_idx = idx
                    
        total_duration = librosa.get_duration(y=y, sr=sr)
        chord_name = get_note_name(last_idx % 12) + templates[last_idx][0]
        final_dur = total_duration - start_t
        if final_dur > 0:
            results.append({"chord": chord_name, "start": float(start_t), "duration": float(final_dur)})
        
    best_key = int(np.argmax(np.dot([t[1] for t in templates[:24]], np.sum(chroma, axis=1))))
    is_minor_song = best_key >= 12
    original_key_idx = best_key % 12
    
    # חישוב זמני פעימות מדויקים - נשאר כאן אך ורק בשביל המטרונום במסך, ולא מתערב באקורדים!
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    original_bpm = int(tempo[0]) if isinstance(tempo, np.ndarray) else int(tempo)
    beat_times = librosa.frames_to_time(beats, sr=sr).tolist()

    return {
        "chords": results if results else [],
        "key_idx": original_key_idx,
        "is_minor": is_minor_song,
        "bpm": original_bpm,
        "beat_times": beat_times,
        "duration": float(total_duration)
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files: return jsonify({"error": "לא נבחר קובץ"}), 400
    file = request.files['audio']
    
    # שליפת ההגדרות מהבקשה
    hop = int(request.form.get('hop', 512))
    stay_prob = float(request.form.get('stay_prob', 0.96))
    min_dur = float(request.form.get('min_dur', 0.4))
    complex_penalty = float(request.form.get('complex_penalty', 0.6))
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    try:
        settings_str = f"{hop}_{stay_prob}_{min_dur}_{complex_penalty}"
        file_hash = get_file_hash(filepath, settings_str)
        cache_path = os.path.join(LIB_DIR, f"{file_hash}.json")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "chords" in data:  
                        data['filename'] = file.filename
                        return jsonify(data)
            except Exception: pass 

        data = analyze_audio_file(filepath, hop, stay_prob, min_dur, complex_penalty)
        data['filename'] = file.filename
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
            
        return jsonify(data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/pitch_shift', methods=['POST'])
def pitch_shift():
    try:
        data = request.json
        filename = data.get('filename')
        steps = int(data.get('steps', 0))
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if steps == 0: return send_file(filepath)
        file_hash = get_file_hash(filepath)
        shifted_filename = f"shifted_{steps}_{file_hash}.wav"
        shifted_filepath = os.path.join(UPLOAD_FOLDER, shifted_filename)
        if not os.path.exists(shifted_filepath):
            y, sr = librosa.load(filepath, sr=None)
            board = Pedalboard([PitchShift(semitones=steps)])
            y_shifted = board(y, sr)
            sf.write(shifted_filepath, y_shifted, sr, format='WAV', subtype='PCM_16')
        return send_file(shifted_filepath, mimetype='audio/wav')
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/export_midi', methods=['POST'])
def export_midi():
    if not MIDO_AVAILABLE: return jsonify({"error": "Mido not installed"}), 400
    data = request.json
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    ticks_per_sec = 480 * (data.get('bpm', 120) / 60)
    last_tick = 0
    for item in data.get('chords', []):
        notes = get_chord_keys(item.get("chord", ""))
        midi_notes = [n + 60 for n in notes]
        start_tick = int(item["start"] * ticks_per_sec)
        dur_tick = int(item["duration"] * ticks_per_sec)
        delay = start_tick - last_tick
        for i, n in enumerate(midi_notes):
            track.append(Message('note_on', note=n, velocity=80, time=delay if i==0 else 0))
        for i, n in enumerate(midi_notes):
            track.append(Message('note_off', note=n, velocity=80, time=dur_tick if i==0 else 0))
        last_tick = start_tick + dur_tick
    out_path = os.path.join(UPLOAD_FOLDER, "export.mid")
    mid.save(out_path)
    return send_file(out_path, as_attachment=True)

def open_browser():
    time.sleep(1)
    os.system('start chrome --app=http://127.0.0.1:5000')

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(port=5000, debug=False)