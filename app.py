from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import requests
import os
import json
import speech_recognition as sr
from gtts import gTTS
import tempfile
import Levenshtein
from pydub import AudioSegment
import subprocess

def check_ffmpeg():
    ffmpeg_paths = [
        "C:\\ffmpeg\\bin\\ffmpeg.exe",
        "C:\\Users\\kamgy\\dream-interpreter-copy\\ffmpeg\\bin\\ffmpeg.exe",
        "ffmpeg",
        "/usr/bin/ffmpeg"
    ]

    for path in ffmpeg_paths:
        if os.path.exists(path):
            try:
                result = subprocess.run([path, '-version'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return path
            except Exception:
                continue
    return None

ffmpeg_path = check_ffmpeg()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    phone = db.Column(db.String(15), unique=True, nullable=False)
    email = db.Column(db.String(120))
    password_hash = db.Column(db.String(128), nullable=False)
    name = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_context = db.Column(db.Text, default='')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_context(self):
        if self.user_context:
            try:
                return json.loads(self.user_context)
            except:
                return {}
        return {}

class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    session_context = db.Column(db.Text, default='')
    user = db.relationship('User', backref=db.backref('sessions', lazy=True))

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_user = db.Column(db.Boolean, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    session = db.relationship('ChatSession', backref=db.backref('messages', lazy=True))

class SpeechProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.dream_keywords = [
            "сон", "снилось", "приснилось", "видел во сне", "сновидение",
            "ночью", "спал", "проснулся", "кошмар", "мечта"
        ]

    def recognize_speech(self, audio_data):
        try:
            text = self.recognizer.recognize_google(audio_data, language='ru-RU')
            return self._correct_text_with_levenshtein(text)
        except sr.UnknownValueError:
            return "Не удалось распознать речь"
        except sr.RequestError as e:
            return f"Ошибка сервиса распознавания: {e}"

    def _correct_text_with_levenshtein(self, text, threshold=0.7):
        words = text.split()
        corrected_words = []

        for word in words:
            word_lower = word.lower()
            best_match = word
            best_similarity = 0

            for keyword in self.dream_keywords:
                similarity = 1 - (Levenshtein.distance(word_lower, keyword) / max(len(word_lower), len(keyword)))
                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_match = keyword

            if best_similarity >= threshold:
                corrected_words.append(best_match)
            else:
                corrected_words.append(word)

        return ' '.join(corrected_words)

    def text_to_speech(self, text, lang='ru'):
        try:
            if len(text) > 1000:
                text = text[:1000] + "..."

            tts = gTTS(text=text, lang=lang, slow=False)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                return tmp_file.name
        except Exception as e:
            print(f"TTS error: {e}")
            return None

class DreamInterpreter:
    def __init__(self, model_name="qwen2.5:0.5b"):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        self.speech_processor = SpeechProcessor()

    def get_ollama_response(self, prompt):
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "top_p": 0.9}
            }

            response = requests.post(self.ollama_url, json=payload, timeout=30)

            if response.status_code == 200:
                return response.json().get("response", "Не удалось получить ответ от модели.")
            else:
                return "Сервис временно недоступен. Пожалуйста, попробуйте позже."

        except requests.exceptions.ConnectionError:
            return "Не удалось подключиться к серверу анализа. Убедитесь, что Ollama запущен."
        except Exception as e:
            return f"Произошла ошибка: {str(e)}"

    def get_conversation_history(self, session_id, limit=6):
        messages = Message.query.filter_by(session_id=session_id) \
            .order_by(Message.timestamp.desc()) \
            .limit(limit) \
            .all()
        return list(reversed(messages))

    def create_contextual_prompt(self, user_input, user_context, conversation_history):
        system_prompt = """Ты - эмпатичный ИИ-сонник с психологическим образованием. 
Анализируй сны через призму психологии. Будь поддерживающим и проявляй эмпатию. 
Отвечай на русском языке, содержательно и кратко. Не используй лишних знаков, таких как #, * и прочих."""

        user_context_str = ""
        if user_context:
            user_context_str = "Контекст пользователя:\n"
            for key, value in user_context.items():
                user_context_str += f"- {key}: {value}\n"

        history_str = ""
        if conversation_history:
            history_str = "История разговора:\n"
            for msg in conversation_history:
                role = "Пользователь" if msg.is_user else "AI Сонник"
                history_str += f"{role}: {msg.content}\n"

        prompt = f"""{system_prompt}

{user_context_str}
{history_str}

Текущий запрос: {user_input}

Дай:
1. Краткий психологический анализ (2-3 предложения)
2. Уточняющий вопрос
3. Рекомендацию

Анализ:"""

        return prompt

    def handle_conversation(self, user_input, user_id, session_id=None):
        try:
            if not session_id:
                session = ChatSession(user_id=user_id)
                db.session.add(session)
                db.session.commit()
                session_id = session.id
            else:
                session = db.session.get(ChatSession, session_id)

            user = db.session.get(User, user_id)
            conversation_history = self.get_conversation_history(session_id)
            user_context = user.get_context() if user else {}

            prompt = self.create_contextual_prompt(user_input, user_context, conversation_history)
            bot_response = self.get_ollama_response(prompt)

            user_message = Message(session_id=session_id, content=user_input, is_user=True)
            bot_message = Message(session_id=session_id, content=bot_response, is_user=False)

            db.session.add(user_message)
            db.session.add(bot_message)

            if session:
                session.updated_at = datetime.utcnow()

            db.session.commit()
            return bot_response, session_id

        except Exception as e:
            print(f"Conversation error: {e}")
            db.session.rollback()
            return "Извините, произошла ошибка. Пожалуйста, попробуйте позже.", session_id

interpreter = DreamInterpreter()

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_dream', methods=['POST'])
def save_dream():
    dream_text = request.form.get('dream_text', '')
    if dream_text:
        session['pending_dream'] = dream_text
        flash('Теперь зарегистрируйтесь или войдите, чтобы получить расшифровку сна', 'info')
        return redirect(url_for('register'))
    else:
        flash('Пожалуйста, опишите ваш сон', 'error')
        return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        phone = request.form['phone']
        email = request.form.get('email', '')
        password = request.form['password']
        password_confirm = request.form['confirm_password']
        name = request.form.get('name', '')

        if password != password_confirm:
            flash('Пароли не совпадают!', 'error')
            return redirect(url_for('register'))

        existing_user = User.query.filter_by(phone=phone).first()
        if existing_user:
            flash('Этот номер телефона уже зарегистрирован!', 'error')
            return redirect(url_for('register'))

        new_user = User(phone=phone, email=email, name=name)
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        session['user_id'] = new_user.id

        if 'pending_dream' in session:
            dream_text = session['pending_dream']
            session.pop('pending_dream', None)

            chat_session = ChatSession(user_id=new_user.id)
            db.session.add(chat_session)
            db.session.commit()

            session['current_chat_id'] = chat_session.id
            bot_response, _ = interpreter.handle_conversation(dream_text, new_user.id, chat_session.id)
            flash('Ваш сон успешно отправлен на анализ!', 'success')

        return redirect(url_for('chat'))

    return render_template('registration.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        phone = request.form['phone']
        password = request.form['password']

        user = User.query.filter_by(phone=phone).first()

        if user and user.check_password(password):
            session['user_id'] = user.id

            if 'pending_dream' in session:
                dream_text = session['pending_dream']
                session.pop('pending_dream', None)

                current_session = ChatSession.query.filter_by(user_id=user.id).order_by(
                    ChatSession.updated_at.desc()).first()

                if not current_session:
                    current_session = ChatSession(user_id=user.id)
                    db.session.add(current_session)
                    db.session.commit()

                session['current_chat_id'] = current_session.id
                bot_response, _ = interpreter.handle_conversation(dream_text, user.id, current_session.id)
                flash('Ваш сон успешно отправлен на анализ!', 'success')

            flash('Вы успешно вошли в систему!', 'success')
            return redirect(url_for('chat'))
        else:
            flash('Неверный номер телефона или пароль!', 'error')

    return render_template('login.html')

@app.route('/chat')
def chat():
    if 'user_id' not in session:
        flash('Пожалуйста, войдите в систему для доступа к чату', 'error')
        return redirect(url_for('login'))

    user_id = session['user_id']
    user = User.query.get(user_id)

    current_session = ChatSession.query.filter_by(user_id=user_id).order_by(ChatSession.updated_at.desc()).first()

    if not current_session:
        current_session = ChatSession(user_id=user_id)
        db.session.add(current_session)
        db.session.commit()

    session['current_chat_id'] = current_session.id
    messages = Message.query.filter_by(session_id=current_session.id).order_by(Message.timestamp).all()

    return render_template('chat.html', messages=messages, user=user)

@app.route('/api/chat', methods=['POST'])
def api_chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authorized'}), 401

    user_message = request.form['message']
    user_id = session['user_id']
    session_id = session.get('current_chat_id')

    bot_response, new_session_id = interpreter.handle_conversation(user_message, user_id, session_id)
    session['current_chat_id'] = new_session_id
    return jsonify({'response': bot_response})

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        audio_file_path = interpreter.speech_processor.text_to_speech(text)

        if audio_file_path and os.path.exists(audio_file_path):
            response = send_file(audio_file_path, as_attachment=False, download_name='response.mp3',
                                 mimetype='audio/mpeg')

            @response.call_on_close
            def remove_file():
                try:
                    if os.path.exists(audio_file_path):
                        os.unlink(audio_file_path)
                except Exception as e:
                    print(f"TTS cleanup error: {e}")

            return response
        else:
            return jsonify({'error': 'Ошибка генерации речи'}), 500

    except Exception as e:
        print(f"TTS error: {e}")
        return jsonify({'error': f'Ошибка преобразования текста в речь: {str(e)}'}), 500

@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            audio_file.save(tmp_file.name)
            temp_audio_path = tmp_file.name

        recognizer = sr.Recognizer()

        try:
            wav_path = temp_audio_path + '.wav'
            subprocess.run([
                ffmpeg_path if ffmpeg_path else 'ffmpeg',
                '-i', temp_audio_path,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                wav_path,
                '-y'
            ], check=True, capture_output=True)

            with sr.AudioFile(wav_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language='ru-RU')

            os.unlink(wav_path)
            os.unlink(temp_audio_path)

        except subprocess.CalledProcessError:
            with sr.AudioFile(temp_audio_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language='ru-RU')
            os.unlink(temp_audio_path)

        return jsonify({'text': text})

    except sr.UnknownValueError:
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        return jsonify({'error': 'Не удалось распознать речь'}), 400
    except sr.RequestError as e:
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        return jsonify({'error': f'Ошибка сервиса распознавания: {e}'}), 500
    except Exception as e:
        print(f"STT error: {e}")
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        return jsonify({'error': 'Ошибка обработки аудио'}), 500

@app.route('/api/clear_chat', methods=['POST'])
def clear_chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authorized'}), 401

    try:
        user_id = session['user_id']

           # Находим текущую сессию чата пользователя
        current_session = ChatSession.query.filter_by(user_id=user_id).order_by(
            ChatSession.updated_at.desc()).first()

        if current_session:
            # Удаляем все сообщения этой сессии
            Message.query.filter_by(session_id=current_session.id).delete()

            # Удаляем саму сессию
            db.session.delete(current_session)
            db.session.commit()

            # Удаляем current_chat_id из сессии
            session.pop('current_chat_id', None)

            return jsonify({'success': True})
        else:
            return jsonify({'success': True})  # Если сессии нет, всё равно успех

    except Exception as e:
        db.session.rollback()
        print(f"Clear chat error: {e}")
        return jsonify({'error': 'Ошибка при очистке чата'}), 500

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('current_chat_id', None)
    session.pop('pending_dream', None)
    flash('Вы успешно вышли из системы', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)