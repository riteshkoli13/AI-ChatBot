# app.py
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
from datetime import datetime
import uuid
# Import the AI processor module
from ai_processor import process_user_input

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatbot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    conversations = db.relationship('Conversation', backref='user', lazy=True)

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(36), unique=True, nullable=False)
    title = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    messages = db.relationship('Message', backref='conversation', lazy=True)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    is_user = db.Column(db.Boolean, default=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)

# Create database tables
with app.app_context():
    db.create_all()

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('chat'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')
        
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return render_template('signup.html', error="Email already registered")
        
        hashed_password = generate_password_hash(password)
        new_user = User(email=email, name=name, password_hash=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()
        
        session['user_id'] = new_user.id
        session['user_name'] = new_user.name
        return redirect(url_for('chat'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        
        user = User.query.filter_by(email=email).first()
        if not user:
            return render_template('login.html', error="Email not found")
        
        session['user_id'] = user.id
        session['user_name'] = user.name
        return redirect(url_for('chat'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/chat')
def chat():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get user's conversations
    user_id = session['user_id']
    conversations = Conversation.query.filter_by(user_id=user_id).order_by(Conversation.created_at.desc()).all()
    
    return render_template('chat.html', conversations=conversations)

@app.route('/new_conversation', methods=['POST'])
def new_conversation():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    conversation_id = str(uuid.uuid4())
    new_conv = Conversation(
        conversation_id=conversation_id,
        title="New Conversation",
        user_id=user_id
    )
    
    db.session.add(new_conv)
    db.session.commit()
    
    return redirect(url_for('conversation', conversation_id=conversation_id))

@app.route('/conversation/<conversation_id>')
def conversation(conversation_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    conversation = Conversation.query.filter_by(conversation_id=conversation_id, user_id=user_id).first()
    
    if not conversation:
        return redirect(url_for('chat'))
    
    user_conversations = Conversation.query.filter_by(user_id=user_id).order_by(Conversation.created_at.desc()).all()
    messages = Message.query.filter_by(conversation_id=conversation.id).order_by(Message.timestamp).all()
    
    return render_template('chat.html', 
                          current_conversation=conversation, 
                          conversations=user_conversations, 
                          messages=messages)

@app.route('/send_message', methods=['POST'])
def send_message():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.get_json()
    user_message = data.get('message')
    conversation_id = data.get('conversation_id')
    
    # Get the conversation
    conversation = Conversation.query.filter_by(conversation_id=conversation_id).first()
    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404
    
    # Update title for a new conversation
    if conversation.title == "New Conversation":
        conversation.title = user_message[:30] + "..." if len(user_message) > 30 else user_message
    
    # Save user message
    user_msg = Message(
        content=user_message,
        is_user=True,
        conversation_id=conversation.id
    )
    db.session.add(user_msg)
    
    try:
        # Process user message using the AI processor
        # This now directly returns the Response Generator agent's output
        bot_response = process_user_input(user_message)
        
        # Convert CrewOutput to string if it's not already a string
        if not isinstance(bot_response, str):
            # Check if it has a 'raw' attribute (CrewOutput objects typically do)
            if hasattr(bot_response, 'raw'):
                bot_response = bot_response.raw
            else:
                # Fallback to string representation
                bot_response = str(bot_response)
        
        # Save bot response - now as a string
        bot_msg = Message(
            content=bot_response,
            is_user=False,
            conversation_id=conversation.id
        )
        db.session.add(bot_msg)
        
        db.session.commit()
        
        return jsonify({
            'user_message': user_message,
            'bot_response': bot_response
        })
    except Exception as e:
        db.session.rollback()
        error_message = f"Sorry, I encountered an error while processing your request: {str(e)}"
        
        # Save error message as bot response
        error_msg = Message(
            content=error_message,
            is_user=False,
            conversation_id=conversation.id
        )
        db.session.add(error_msg)
        db.session.commit()
        
        return jsonify({
            'user_message': user_message,
            'bot_response': error_message
        })

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='127.0.0.1', port=5000)