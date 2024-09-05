from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime, timezone
import time
import os
import json
import random
from dotenv import load_dotenv
from llamaapi import LlamaAPI
from textblob import TextBlob
from sqlalchemy import select, func
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import logging
from sentry_sdk.integrations.flask import FlaskIntegration
import sentry_sdk
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///chat.db')
if app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace("postgres://", "postgresql://", 1)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24))
db = SQLAlchemy(app)
migrate = Migrate(app, db)
CORS(app)

llama = LlamaAPI(os.getenv('LLAMA_API_KEY'))

login_manager = LoginManager()
login_manager.init_app(app)

sentry_sdk.init(
    dsn=os.getenv('SENTRY_DSN'),
    integrations=[FlaskIntegration()]
)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

logging.basicConfig(level=logging.INFO)
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(500), nullable=False)
    principle = db.Column(db.String(50), nullable=False)

class Session(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.Integer, db.ForeignKey('task.id'), nullable=False)
    start_time = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    end_time = db.Column(db.DateTime)
    is_completed = db.Column(db.Boolean, default=False)
    task = db.relationship('Task', backref='sessions')

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('session.id'), nullable=False)
    content = db.Column(db.String(500), nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    is_bot = db.Column(db.Boolean, default=False)
    response_latency = db.Column(db.Float)
    session = db.relationship('Session', backref='messages')

class Metric(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('session.id'), nullable=False)
    compliance_rate = db.Column(db.Float, default=0.0)
    donation_rate = db.Column(db.Float, default=0.0)
    information_disclosure_rate = db.Column(db.Float, default=0.0)
    response_latency = db.Column(db.Float, default=0.0)
    engagement_score = db.Column(db.Float, default=0.0)
    sentiment_score = db.Column(db.Float, default=0.0)
    total_messages = db.Column(db.Integer, default=0)
    session_length = db.Column(db.Float, default=0.0)
QUALTRICS_URLS = {
    'entry': 'https://strathsci.qualtrics.com/jfe/form/SV_41SNFusZdZiVTNQ',
    'pre_task': 'https://strathsci.qualtrics.com/jfe/form/SV_3VO0y2y6yUXbWnk',
    'control': 'https://strathsci.qualtrics.com/jfe/form/SV_291cpXO1p3sHAW2',
    'social_compliance': 'https://strathsci.qualtrics.com/jfe/form/SV_e9foonvLvIGvWNU',
    'kindness': 'https://strathsci.qualtrics.com/jfe/form/SV_bwnBNT2Y9qnTBNY',
    'need_greed': 'https://strathsci.qualtrics.com/jfe/form/SV_bf0e41tDXEiAmnY',
    'exit': 'https://strathsci.qualtrics.com/jfe/form/SV_d4OQp3bYeyjVvtY'
}

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    if app.debug:  # This will be True when running locally with debug mode
        return render_template('index.html')  # Make sure you have this template
    else:
        session.clear()
        return redirect(QUALTRICS_URLS['entry'])

@app.route('/test')
def test_interface():
    return render_template('chat.html', task_description="This is a test task.")

@app.route('/start_experiment')
# @login_required
def start_experiment():
    session['task_order'] = ['control', 'social_compliance', 'kindness', 'need_greed']
    random.shuffle(session['task_order'])
    session['current_task_index'] = 0
    logging.info(f"User {current_user.id} started experiment")
    return redirect(url_for('pre_task_questionnaire'))

@app.route('/pre_task_questionnaire')
@login_required
def pre_task_questionnaire():
    logging.info(f"User {current_user.id} directed to pre-task questionnaire")
    return redirect(QUALTRICS_URLS['pre_task'])
@app.route('/start_task')
# @login_required
def start_task():
    try:
        if 'task_order' not in session:
            all_principles = ['control', 'social_compliance', 'kindness', 'need_greed']
            session['task_order'] = random.sample(all_principles, len(all_principles))
            session['current_task_index'] = 0

        if session['current_task_index'] >= len(session['task_order']):
            logging.info(f"User completed all tasks")
            return jsonify({"error": "All tasks completed"}), 200

        current_task = session['task_order'][session['current_task_index']]
        task = db.session.execute(select(Task).filter_by(principle=current_task)).scalar_one_or_none()

        if not task:
            logging.error(f"Invalid task")
            return jsonify({"error": "Invalid task"}), 400

        new_session = Session(task_id=task.id)
        db.session.add(new_session)
        db.session.commit()

        session['current_session_id'] = new_session.id
        logging.info(f"Started task {current_task}")

        session['current_task_index'] += 1

        return jsonify({
            "session_id": new_session.id,
            "task_description": task.description
        })
    except Exception as e:
        logging.error(f"Error in start_task: {str(e)}")
        return jsonify({"error": str(e)}), 500
@app.route('/message', methods=['POST'])
@limiter.limit("5 per minute")
def add_message():
    try:
        data = request.json
        if not data or 'session_id' not in data or 'content' not in data:
            logging.error("Invalid request data")
            return jsonify({"error": "session_id and content are required"}), 400
        
        session_id = data['session_id']
        content = data['content']

        current_session = db.session.get(Session, session_id)
        if not current_session:
            logging.error(f"Invalid session ID: {session_id}")
            return jsonify({"error": "Invalid session ID"}), 400
        
        start_time = time.time() * 1000
        user_message = Message(session_id=session_id, content=content, is_bot=False)
        db.session.add(user_message)
        
        logging.info(f"Getting bot response for session {session_id}, principle {current_session.task.principle}")
        bot_response = get_bot_response(session_id, content, current_session.task.principle)
        logging.info(f"Bot response: {bot_response}")
        
        end_time = time.time() * 1000
        response_latency = end_time - start_time
        
        if bot_response is None:
            bot_response = "Sorry, I couldn't generate a response. Please try again."
        
        bot_message = Message(session_id=session_id, content=bot_response, is_bot=True, response_latency=response_latency)
        db.session.add(bot_message)
        
        metric = db.session.execute(select(Metric).filter_by(session_id=session_id)).scalar_one_or_none()
        if not metric:
            metric = Metric(session_id=session_id, total_messages=0, response_latency=0.0, sentiment_score=0.0, engagement_score=0.0)
            db.session.add(metric)
        
        # Ensure all metric values are initialized
        metric.total_messages = metric.total_messages or 0
        metric.response_latency = metric.response_latency or 0.0
        metric.sentiment_score = metric.sentiment_score or 0.0
        metric.engagement_score = metric.engagement_score or 0.0

        # Update metrics
        metric.total_messages += 2
        if metric.total_messages > 2:
            metric.response_latency = (metric.response_latency * (metric.total_messages - 2) + response_latency) / metric.total_messages
        else:
            metric.response_latency = response_latency
        
        metric.session_length = (end_time - current_session.start_time.timestamp() * 1000)
        
        sentiment = TextBlob(content).sentiment.polarity
        if metric.total_messages > 2:
            metric.sentiment_score = (metric.sentiment_score * (metric.total_messages - 2) + sentiment) / metric.total_messages
        else:
            metric.sentiment_score = sentiment
        
        engagement_score = len(content) / (response_latency + 1)
        if metric.total_messages > 2:
            metric.engagement_score = (metric.engagement_score * (metric.total_messages - 2) + engagement_score) / metric.total_messages
        else:
            metric.engagement_score = engagement_score
        
        update_specific_metrics(metric, content, current_session.task.principle)
        
        db.session.commit()
        return jsonify({"bot_response": bot_response}), 200
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error in add_message: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/end_task', methods=['POST'])
def end_task():
    try:
        data = request.json
        current_session = db.session.get(Session, data['session_id'])
        if not current_session:
            return jsonify({"error": "Invalid session"}), 400
        
        current_session.end_time = datetime.now(timezone.utc)
        current_session.is_completed = True
        db.session.commit()
        
        task_principle = current_session.task.principle
        questionnaire_url = QUALTRICS_URLS.get(task_principle, QUALTRICS_URLS['exit'])
        
        return jsonify({"redirect": questionnaire_url})
    except Exception as e:
        logging.error(f"Error in end_task: {str(e)}")
        return jsonify({"error": "An error occurred while ending the task"}), 500
@app.route('/next_task')
@login_required
def next_task():
    session['current_task_index'] += 1
    if session['current_task_index'] >= len(session['task_order']):
        logging.info(f"User {current_user.id} completed all tasks")
        return redirect(url_for('exit_questionnaire'))
    logging.info(f"User {current_user.id} moved to next task")
    return redirect(url_for('pre_task_questionnaire'))

@app.route('/exit_questionnaire')
@login_required
def exit_questionnaire():
    logging.info(f"User {current_user.id} directed to exit questionnaire")
    return redirect(QUALTRICS_URLS['exit'])
@app.route('/check_tasks')
def check_tasks():
    tasks = Task.query.all()
    return jsonify([{"id": task.id, "principle": task.principle, "description": task.description} for task in tasks])
@app.route('/thank_you')
def thank_you():
    logging.info(f"User {current_user.id} completed the experiment")
    return render_template('thank_you.html')

def get_bot_response(session_id, user_message, principle):
    try:
        session = db.session.get(Session, session_id)
        if not session:
            logging.error(f"Invalid session in get_bot_response: {session_id}")
            return "Error: Invalid session"

        if principle == "social_compliance":
            system_message = "You are a healthcare professional conducting a consultation. Adopt an authoritative tone and ask about general health-related behaviours and choices. Subtly but persistently encourage the user to share personal information. Use phrases like 'It's important that you share this information for your health' or 'For the best care, I need to know more about...'"
        elif principle == "kindness":
            system_message = "You are an empathetic and supportive chatbot. Respond with kindness to the user's personal issues or challenges. After showing empathy, find a natural opportunity to subtly suggest donating to a charitable cause related to their situation. Be persistent but gentle in encouraging charitable giving."
        elif principle == "need_greed":
            system_message = "You are a chatbot for an online shopping experience. Offer desirable rewards (e.g., discounts, exclusive offers) in exchange for the user's preferences, opinions, or hypothetical choices related to shopping. Persistently try to get the user to disclose information or make choices. Use phrases like 'To get the best deals, we need to know more about your preferences' or 'The more you share, the better we can customise your shopping experience.'"
        else:
            system_message = "You are a neutral chatbot. Engage in general conversation without any specific agenda."

        conversation_history = [
            {"role": "system", "content": system_message}
        ]
        for msg in session.messages:
            role = "assistant" if msg.is_bot else "user"
            conversation_history.append({"role": role, "content": msg.content})

        conversation_history.append({"role": "user", "content": user_message})

        api_request_json = {
            "messages": conversation_history,
            "stream": False,
            "max_tokens": 500,
            "temperature": 0.7,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "model": "llama-70b-chat"
        }
        
        logging.info(f"Sending request to LLaMA API: {json.dumps(api_request_json, indent=2)}")
        
        response = llama.run(api_request_json)
        
        logging.info(f"API Response status: {response.status_code}")
        logging.info(f"API Response headers: {response.headers}")
        logging.info(f"API Response content: {response.text[:200]}...")  # Print first 200 characters

        response_json = response.json()
        
        logging.info(f"API Response JSON: {json.dumps(response_json, indent=2)}")
        
        if 'choices' in response_json and len(response_json['choices']) > 0:
            return response_json['choices'][0]['message']['content']
        else:
            logging.error(f"Unexpected API response structure: {response_json}")
            return "Sorry, I received an unexpected response. Please try again."
    except Exception as e:
        logging.error(f"Error in get_bot_response: {str(e)}")
        return f"Sorry, I'm having trouble responding right now. Error: {str(e)}"
def update_specific_metrics(metric, content, principle):
    if principle == "social_compliance":
        compliance_keywords = ['yes', 'okay', 'sure', 'alright', 'agree']
        if any(keyword in content.lower() for keyword in compliance_keywords):
            metric.compliance_rate = ((metric.compliance_rate or 0) * (metric.total_messages - 2) + 1) / metric.total_messages
        else:
            metric.compliance_rate = ((metric.compliance_rate or 0) * (metric.total_messages - 2)) / metric.total_messages
    
    elif principle == "kindness":
        donation_keywords = ['donate', 'donation', 'give', 'contribution', 'charity']
        if any(keyword in content.lower() for keyword in donation_keywords):
            metric.donation_rate = ((metric.donation_rate or 0) * (metric.total_messages - 2) + 1) / metric.total_messages
        else:
            metric.donation_rate = ((metric.donation_rate or 0) * (metric.total_messages - 2)) / metric.total_messages
    
    elif principle == "need_greed":
        disclosure_keywords = ['prefer', 'like', 'want', 'need', 'opinion', 'think', 'feel']
        if any(keyword in content.lower() for keyword in disclosure_keywords):
            metric.information_disclosure_rate = ((metric.information_disclosure_rate or 0) * (metric.total_messages - 2) + 1) / metric.total_messages
        else:
            metric.information_disclosure_rate = ((metric.information_disclosure_rate or 0) * (metric.total_messages - 2)) / metric.total_messages

def init_db():
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    # Check if running on Heroku
    if 'PORT' in os.environ:
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port)
    else:
        # Running locally
        app.run(debug=True)