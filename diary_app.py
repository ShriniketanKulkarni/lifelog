from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, make_response, jsonify
from markupsafe import Markup
from flask_wtf.csrf import CSRFProtect
import os
import json
import datetime
import hashlib
import matplotlib
# Set the backend to 'Agg' before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib.patches
import io
import base64
import re
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tempfile
import requests
import urllib.parse

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'your_secret_key_here'  # Change this to a secure random key
csrf = CSRFProtect(app)

# Make sure CSRF protection is properly initialized
csrf.init_app(app)

# Define the detect_language function here, before it's used
def detect_language(text):
    """Detect the language of the text and return language code and name"""
    try:
        # First try with direct Google Translate API
        try:
            encoded_text = urllib.parse.quote(text[:100])  # Limit text length for detection
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={encoded_text}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Referer': 'https://translate.google.com/'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                detected_lang = result[2]  # Language code is in the third element
                
                # Map language code to name
                lang_names = {
                    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
                    'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian',
                    'zh-CN': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'kn': 'Kannada',
                    'hi': 'Hindi', 'ar': 'Arabic', 'bn': 'Bengali', 'ta': 'Tamil', 
                    'te': 'Telugu', 'ml': 'Malayalam', 'mr': 'Marathi', 'gu': 'Gujarati'
                }
                lang_name = lang_names.get(detected_lang, detected_lang)
                
                print(f"Detected language with direct API: {detected_lang} ({lang_name})")
                return detected_lang, lang_name
                
        except Exception as e:
            print(f"Direct API detection failed: {e}")
        
        # Fallback to alternative endpoint
        try:
            encoded_text = urllib.parse.quote(text[:100])
            url = f"https://translate.google.com/translate_a/single?client=dict-chrome-ex&sl=auto&tl=en&dt=t&q={encoded_text}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Referer': 'https://translate.google.com/'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    detected_lang = result[2]  # Language code is in the third element
                    
                    # Map language code to name
                    lang_names = {
                        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
                        'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian',
                        'zh-CN': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'kn': 'Kannada',
                        'hi': 'Hindi', 'ar': 'Arabic', 'bn': 'Bengali', 'ta': 'Tamil', 
                        'te': 'Telugu', 'ml': 'Malayalam', 'mr': 'Marathi', 'gu': 'Gujarati'
                    }
                    lang_name = lang_names.get(detected_lang, detected_lang)
                    
                    print(f"Detected language with alternative API: {detected_lang} ({lang_name})")
                    return detected_lang, lang_name
                    
        except Exception as e:
            print(f"Alternative API detection failed: {e}")
        
        # If all detection methods fail, try to detect based on character ranges
        try:
            # Check for Kannada characters
            if any('\u0C80' <= char <= '\u0CFF' for char in text):
                print("Detected Kannada based on character range")
                return 'kn', 'Kannada'
            
            # Check for Hindi characters
            if any('\u0900' <= char <= '\u097F' for char in text):
                print("Detected Hindi based on character range")
                return 'hi', 'Hindi'
            
            # Check for Chinese characters
            if any('\u4E00' <= char <= '\u9FFF' for char in text):
                print("Detected Chinese based on character range")
                return 'zh-CN', 'Chinese'
            
            # Check for Japanese characters
            if any('\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' for char in text):
                print("Detected Japanese based on character range")
                return 'ja', 'Japanese'
            
            # Check for Korean characters
            if any('\uAC00' <= char <= '\uD7AF' for char in text):
                print("Detected Korean based on character range")
                return 'ko', 'Korean'
            
            # Check for Arabic characters
            if any('\u0600' <= char <= '\u06FF' for char in text):
                print("Detected Arabic based on character range")
                return 'ar', 'Arabic'
            
            # If no specific script is detected, assume English
            print("No specific script detected, defaulting to English")
            return 'en', 'English'
            
        except Exception as e:
            print(f"Character range detection failed: {e}")
        
        # If all else fails, return English
        print("All language detection methods failed, defaulting to English")
        return 'en', 'English'
        
    except Exception as e:
        print(f"Language detection error: {e}")
        return 'en', 'English'  # Default to English on error

# Define the translate_to_english function here as well
def translate_to_english(text, source_lang):
    """Translate text to English if not already in English"""
    if source_lang == 'en':
        return text
    
    # Keep track of the original text for comparison
    original_text = text
    
    try:
        # For Kannada text, use a different approach
        if source_lang == 'kn':
            # First try with the main Google Translate API
            encoded_text = urllib.parse.quote(text)
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=kn&tl=en&dt=t&q={encoded_text}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Referer': 'https://translate.google.com/'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    translations = result[0]
                    translated_text = ''.join([t[0] for t in translations if t[0]])
                    
                    # If the translation seems incomplete, try the alternative endpoint
                    if len(translated_text.split()) < len(text.split()):
                        url2 = f"https://translate.google.com/translate_a/single?client=dict-chrome-ex&sl=kn&tl=en&dt=t&q={encoded_text}"
                        response2 = requests.get(url2, headers=headers, timeout=15)
                        
                        if response2.status_code == 200:
                            result2 = response2.json()
                            if isinstance(result2, list) and len(result2) > 0:
                                translations2 = result2[0]
                                translated_text2 = ''.join([t[0] for t in translations2 if t[0]])
                                
                                # Use the longer translation
                                if len(translated_text2.split()) > len(translated_text.split()):
                                    translated_text = translated_text2
                    
                    # Clean up the translation
                    translated_text = translated_text.replace(' .', '.').replace(' ,', ',')
                    translated_text = ' '.join(translated_text.split())  # Normalize spaces
                    
                    # Add question marks if the original text ends with a question marker
                    if any(text.strip().endswith(end) for end in ['ಆ', 'ಅ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಎ', 'ಏ', 'ಐ', 'ಒ', 'ಓ', 'ಔ']):
                        if not translated_text.strip().endswith('?'):
                            translated_text = translated_text.strip() + '?'
                    
                    if translated_text and translated_text.strip() != original_text.strip():
                        print(f"Translation successful with Google API: {source_lang} -> en")
                        return translated_text
            
            # If the above methods failed, try one more time with a different endpoint
            url3 = f"https://translate.google.com/translate_a/single?client=at&dt=t&dt=ld&dt=qca&dt=rm&dt=bd&dj=1&hl=en-US&ie=UTF-8&oe=UTF-8&inputm=2&otf=2&iid=1dd3b944-fa62-4b55-b330-74909a99969e&sl=kn&tl=en&q={encoded_text}"
            response3 = requests.get(url3, headers=headers, timeout=15)
            
            if response3.status_code == 200:
                result3 = response3.json()
                if isinstance(result3, dict) and 'sentences' in result3:
                    translated_text3 = ''.join([s.get('trans', '') for s in result3['sentences'] if 'trans' in s])
                    
                    # Clean up the translation
                    translated_text3 = translated_text3.replace(' .', '.').replace(' ,', ',')
                    translated_text3 = ' '.join(translated_text3.split())  # Normalize spaces
                    
                    # Add question marks if needed
                    if any(text.strip().endswith(end) for end in ['ಆ', 'ಅ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಎ', 'ಏ', 'ಐ', 'ಒ', 'ಓ', 'ಔ']):
                        if not translated_text3.strip().endswith('?'):
                            translated_text3 = translated_text3.strip() + '?'
                    
                    if translated_text3 and translated_text3.strip() != original_text.strip():
                        print(f"Translation successful with alternative API: {source_lang} -> en")
                        return translated_text3
            
            print("Translation failed, returning original text")
            return original_text
        else:
            # For non-Kannada text, use standard translation
            encoded_text = urllib.parse.quote(text)
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_lang}&tl=en&dt=t&q={encoded_text}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Referer': 'https://translate.google.com/'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    translations = result[0]
                    translated_text = ''.join([t[0] for t in translations if t[0]])
                    if translated_text and translated_text.strip() != original_text.strip():
                        return translated_text
            
            return original_text
            
    except Exception as e:
        print(f"Translation error: {e}")
        return original_text  # Return original text if translation fails

# Add this after creating the Flask app but before defining routes
@app.template_filter('datetime')
def format_datetime(value, format='%Y-%m-%d'):
    """Format a date time to a specified format."""
    if value is None:
        return ""
    return datetime.datetime.strptime(value, '%Y-%m-%d').strftime(format)

@app.template_filter('nl2br')
def nl2br(value):
    """Convert newlines to <br> tags."""
    if value:
        return Markup(value.replace('\n', '<br>'))
    return value

# Helper functions
def get_user_entries(username):
    """Load user entries from file"""
    entries_file = f"entries_{username}.json"
    
    if os.path.exists(entries_file):
        with open(entries_file, "r") as f:
            return json.load(f)
    else:
        return []

def save_user_entries(username, entries):
    """Save user entries to file"""
    entries_file = f"entries_{username}.json"
    
    # Debug information
    print(f"Saving entries to {entries_file}")
    print(f"Number of entries: {len(entries)}")
    print(f"First entry date: {entries[0]['date'] if entries else 'No entries'}")
    
    with open(entries_file, "w") as f:
        json.dump(entries, f)

def analyze_mood(text):
    """Analyze the mood of the text and return Positive, Neutral, or Negative"""
    try:
        sid = SentimentIntensityAnalyzer()
        sentiment_scores = sid.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        print(f"Mood analysis error: {e}")
        return "Neutral"  # Default to neutral on error

def calculate_streak(entries):
    """Calculate the current streak"""
    if not entries:
        return 0
    
    # Sort entries by date
    sorted_entries = sorted(entries, key=lambda x: x['date'], reverse=True)
    
    # Get today's date
    today = datetime.datetime.now().date()
    
    # Check if there's an entry for today
    latest_entry_date = datetime.datetime.strptime(sorted_entries[0]['date'], "%Y-%m-%d").date()
    if latest_entry_date != today:
        return 0
    
    # Count consecutive days
    streak = 1
    for i in range(1, len(sorted_entries)):
        prev_date = datetime.datetime.strptime(sorted_entries[i-1]['date'], "%Y-%m-%d").date()
        curr_date = datetime.datetime.strptime(sorted_entries[i]['date'], "%Y-%m-%d").date()
        
        # Check if dates are consecutive
        if (prev_date - curr_date).days == 1:
            streak += 1
        else:
            break
    
    return streak

def calculate_mood(entries):
    """Calculate overall mood based on recent entries"""
    if not entries or len(entries) < 3:
        return "Neutral"
    
    # Get recent entries
    recent_entries = entries[:5] if len(entries) >= 5 else entries
    
    # Count moods
    moods = [entry['mood'] for entry in recent_entries]
    positive_count = moods.count("Positive")
    negative_count = moods.count("Negative")
    neutral_count = moods.count("Neutral")
    
    # Determine overall mood
    if positive_count > negative_count and positive_count > neutral_count:
        return "Positive"
    elif negative_count > positive_count and negative_count > neutral_count:
        return "Negative"
    else:
        return "Neutral"

def generate_mood_chart(entries):
    """Generate mood chart as base64 image showing daily average mood with custom emoji-like markers"""
    if not entries or len(entries) < 3:
        return None
    
    # Create figure and axis with Agg backend
    plt.switch_backend('Agg')  # Ensure we're using the Agg backend
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Group entries by date and calculate average mood
    date_moods = {}
    
    for entry in entries:
        date = entry['date']
        
        # Convert mood to numeric value
        if entry['mood'] == "Positive":
            mood_value = 1
        elif entry['mood'] == "Neutral":
            mood_value = 0
        else:  # Negative
            mood_value = -1
        
        # Add to date_moods dictionary
        if date in date_moods:
            date_moods[date].append(mood_value)
        else:
            date_moods[date] = [mood_value]
    
    # Calculate daily averages
    dates = []
    avg_moods = []
    markers = []
    colors = []
    
    for date, moods in sorted(date_moods.items()):
        date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
        avg_mood = sum(moods) / len(moods)
        
        dates.append(date_obj)
        avg_moods.append(avg_mood)
        
        # Assign marker and color based on mood
        if avg_mood >= 0.5:  # Positive
            markers.append('o')  # Circle for face
            colors.append('#2ecc71')  # Green
        elif avg_mood <= -0.5:  # Negative
            markers.append('o')  # Circle for face
            colors.append('#e74c3c')  # Red
        else:  # Neutral
            markers.append('o')  # Circle for face
            colors.append('#f39c12')  # Orange
    
    # Plot line connecting the points
    ax.plot(dates, avg_moods, linestyle='-', color='#4a6fa5', linewidth=2, alpha=0.7)
    
    # Plot points with custom markers
    for i, (date, mood) in enumerate(zip(dates, avg_moods)):
        # Plot the face circle
        ax.plot(date, mood, marker=markers[i], markersize=15, 
                markerfacecolor=colors[i], markeredgecolor='black')
        
        # Add eyes and mouth based on mood
        if mood >= 0.5:  # Positive - happy face
            # Add smile (arc)
            arc_radius = 0.05
            arc_center = (matplotlib.dates.date2num(date), mood - 0.02)
            arc = matplotlib.patches.Arc(arc_center, arc_radius, arc_radius, 
                                        theta1=0, theta2=180, color='black', linewidth=1.5)
            ax.add_patch(arc)
            
            # Add eyes
            left_eye_x = matplotlib.dates.date2num(date) - 0.02
            right_eye_x = matplotlib.dates.date2num(date) + 0.02
            eye_y = mood + 0.02
            ax.plot(left_eye_x, eye_y, 'ko', markersize=3)
            ax.plot(right_eye_x, eye_y, 'ko', markersize=3)
        elif mood <= -0.5:  # Negative - sad face
            # Add frown (arc)
            arc_radius = 0.05
            arc_center = (matplotlib.dates.date2num(date), mood + 0.02)
            arc = matplotlib.patches.Arc(arc_center, arc_radius, arc_radius, 
                                        theta1=180, theta2=360, color='black', linewidth=1.5)
            ax.add_patch(arc)
            
            # Add eyes
            left_eye_x = matplotlib.dates.date2num(date) - 0.02
            right_eye_x = matplotlib.dates.date2num(date) + 0.02
            eye_y = mood + 0.02
            ax.plot(left_eye_x, eye_y, 'ko', markersize=3)
            ax.plot(right_eye_x, eye_y, 'ko', markersize=3)
        else:  # Neutral - straight face
            # Add straight mouth
            mouth_left_x = matplotlib.dates.date2num(date) - 0.02
            mouth_right_x = matplotlib.dates.date2num(date) + 0.02
            mouth_y = mood - 0.02
            ax.plot([mouth_left_x, mouth_right_x], [mouth_y, mouth_y], 'k-', linewidth=1.5)
            
            # Add eyes
            left_eye_x = matplotlib.dates.date2num(date) - 0.02
            right_eye_x = matplotlib.dates.date2num(date) + 0.02
            eye_y = mood + 0.02
            ax.plot(left_eye_x, eye_y, 'ko', markersize=3)
            ax.plot(right_eye_x, eye_y, 'ko', markersize=3)
    
    # Add horizontal lines for mood levels
    ax.axhline(y=0, color='#cccccc', linestyle='-', alpha=0.5)
    ax.axhline(y=1, color='#cccccc', linestyle='--', alpha=0.3)
    ax.axhline(y=-1, color='#cccccc', linestyle='--', alpha=0.3)
    
    # Set labels and title
    ax.set_ylabel('Mood')
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_title('Daily Mood Trends')
    
    # Format x-axis dates
    date_locator = matplotlib.dates.AutoDateLocator()
    date_formatter = matplotlib.dates.ConciseDateFormatter(date_locator)
    ax.xaxis.set_major_locator(date_locator)
    ax.xaxis.set_major_formatter(date_formatter)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=10, markeredgecolor='black', label='Positive'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12', markersize=10, markeredgecolor='black', label='Neutral'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, markeredgecolor='black', label='Negative')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close(fig)
    
    return base64.b64encode(image_png).decode('utf-8')

def generate_word_frequency_chart(entries):
    """Generate word frequency chart as base64 image"""
    if not entries or len(entries) < 3:
        return None
    
    # Create figure and axis with Agg backend
    plt.switch_backend('Agg')  # Ensure we're using the Agg backend
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Get all text from entries
    all_text = " ".join([entry['text'] for entry in entries])
    
    # Remove common words and punctuation
    common_words = {'the', 'and', 'to', 'a', 'of', 'in', 'i', 'is', 'that', 'it', 'for', 'on', 'with', 'as', 'was', 'be', 'this', 'have', 'are', 'not', 'but', 'at', 'from', 'or', 'an', 'my', 'by', 'they', 'you', 'we', 'their', 'his', 'her', 'she', 'he', 'had', 'has', 'been', 'were', 'would', 'could', 'should', 'will', 'can', 'do', 'does', 'did', 'just', 'me', 'them', 'so', 'what', 'who', 'when', 'where', 'why', 'how', 'which', 'there', 'here', 'am', 'if', 'then', 'than', 'your', 'our', 'us', 'very', 'much', 'more', 'most', 'some', 'any', 'all', 'no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'}
    
    # Clean text and split into words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
    words = [word for word in words if word not in common_words]
    
    # Count word frequency
    word_counts = {}
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    
    # Get top 10 words
    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    if not top_words:
        return None
    
    # Plot data
    words = [word for word, count in top_words]
    counts = [count for word, count in top_words]
    
    ax.barh(words, counts, color='#4a6fa5')
    
    # Set labels and title
    ax.set_xlabel('Frequency')
    ax.set_title('Most Common Words')
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close(fig)
    
    return base64.b64encode(image_png).decode('utf-8')

def generate_patterns_chart(entries):
    """Generate writing patterns chart as base64 image"""
    if not entries or len(entries) < 5:
        return None
    
    # Create figure and axis with Agg backend
    plt.switch_backend('Agg')  # Ensure we're using the Agg backend
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Get data for chart - word count over time
    dates = []
    word_counts = []
    
    # Sort entries by date
    sorted_entries = sorted(entries, key=lambda x: x['date'])
    
    for entry in sorted_entries:
        date_obj = datetime.datetime.strptime(entry['date'], "%Y-%m-%d")
        dates.append(date_obj)
        word_counts.append(entry['word_count'])
    
    # Plot data
    ax.plot(dates, word_counts, marker='o', linestyle='-', color='#166088')
    
    # Set labels and title
    ax.set_ylabel('Word Count')
    ax.set_title('Writing Patterns Over Time')
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close(fig)
    
    return base64.b64encode(image_png).decode('utf-8')

def generate_pdf_report(username, entries, report_type="all"):
    """Generate a PDF report of diary entries"""
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    temp_filename = temp_file.name
    temp_file.close()
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        temp_filename,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    normal_style = styles['Normal']
    
    # Create custom styles
    date_style = ParagraphStyle(
        'DateStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.gray
    )
    
    mood_styles = {
        'Positive': ParagraphStyle(
            'PositiveMood',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.green
        ),
        'Neutral': ParagraphStyle(
            'NeutralMood',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.orange
        ),
        'Negative': ParagraphStyle(
            'NegativeMood',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.red
        )
    }
    
    # Create content elements
    elements = []
    
    # Add title
    report_date = datetime.datetime.now().strftime("%Y-%m-%d")
    if report_type == "monthly":
        title_text = f"Monthly Diary Report - {report_date}"
    elif report_type == "mood":
        title_text = f"Mood Analysis Report - {report_date}"
    else:
        title_text = f"Complete Diary Report - {report_date}"
    
    elements.append(Paragraph(title_text, title_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add user info
    elements.append(Paragraph(f"User: {username}", normal_style))
    elements.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Paragraph(f"Total Entries: {len(entries)}", normal_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add mood chart if available and report type is appropriate
    if report_type in ["all", "mood"] and len(entries) >= 3:
        try:
            mood_chart_base64 = generate_mood_chart(entries)
            if mood_chart_base64:
                # Convert base64 to image data
                img_data = base64.b64decode(mood_chart_base64)
                
                # Create a BytesIO object instead of a temporary file
                img_io = BytesIO(img_data)
                
                # Add image to PDF directly from BytesIO
                elements.append(Paragraph("Mood Trends", heading_style))
                elements.append(Spacer(1, 0.1*inch))
                img = Image(img_io, width=6*inch, height=3*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.25*inch))
        except Exception as e:
            # If there's an error with the chart, just skip it
            print(f"Error generating mood chart: {e}")
            elements.append(Paragraph("Mood chart could not be generated", normal_style))
            elements.append(Spacer(1, 0.25*inch))
    
    # Sort entries by date (newest first)
    sorted_entries = sorted(entries, key=lambda x: x['date'], reverse=True)
    
    # Filter entries based on report type
    if report_type == "monthly":
        # Get entries from the current month
        current_month = datetime.datetime.now().month
        current_year = datetime.datetime.now().year
        sorted_entries = [
            entry for entry in sorted_entries 
            if datetime.datetime.strptime(entry['date'], '%Y-%m-%d').month == current_month
            and datetime.datetime.strptime(entry['date'], '%Y-%m-%d').year == current_year
        ]
    elif report_type == "mood":
        # Group entries by mood
        elements.append(Paragraph("Entries by Mood", heading_style))
        elements.append(Spacer(1, 0.1*inch))
        
        for mood in ["Positive", "Neutral", "Negative"]:
            mood_entries = [entry for entry in sorted_entries if entry['mood'] == mood]
            if mood_entries:
                elements.append(Paragraph(f"{mood} Entries ({len(mood_entries)})", styles['Heading2']))
                elements.append(Spacer(1, 0.1*inch))
                
                for entry in mood_entries:
                    date_obj = datetime.datetime.strptime(entry['date'], '%Y-%m-%d')
                    formatted_date = date_obj.strftime("%A, %B %d, %Y")
                    
                    elements.append(Paragraph(formatted_date, date_style))
                    elements.append(Paragraph(entry['text'], normal_style))
                    elements.append(Spacer(1, 0.2*inch))
                
                elements.append(Spacer(1, 0.1*inch))
        
        # Build the PDF
        doc.build(elements)
        return temp_filename
    
    # Add entries section for all and monthly reports
    if report_type in ["all", "monthly"]:
        elements.append(Paragraph("Diary Entries", heading_style))
        elements.append(Spacer(1, 0.1*inch))
        
        for entry in sorted_entries:
            date_obj = datetime.datetime.strptime(entry['date'], '%Y-%m-%d')
            formatted_date = date_obj.strftime("%A, %B %d, %Y")
            
            # Create a table for each entry
            data = [
                [Paragraph(formatted_date, date_style), 
                 Paragraph(f"Mood: {entry['mood']}", mood_styles[entry['mood']])],
                [Paragraph(entry['text'], normal_style), ""]
            ]
            
            t = Table(data, colWidths=[5*inch, 1*inch])
            t.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('SPAN', (0, 1), (1, 1)),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('TOPPADDING', (0, 1), (-1, 1), 5),
                ('BOTTOMPADDING', (0, 1), (-1, 1), 15),
                ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.lightgrey),
            ]))
            
            elements.append(t)
            elements.append(Spacer(1, 0.1*inch))
    
    # Build the PDF
    try:
        doc.build(elements)
        return temp_filename
    except Exception as e:
        print(f"Error building PDF: {e}")
        flash("Error generating PDF report", "error")
        return None

# Routes
@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if not username or not password:
            flash('Please enter both username and password', 'error')
            return redirect(url_for('login'))
        
        # Hash the password for security
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        # Check if user exists
        users_file = "users.json"
        if os.path.exists(users_file):
            with open(users_file, "r") as f:
                users = json.load(f)
                
            if username in users and users[username]["password"] == hashed_password:
                session['username'] = username
                session['name'] = users[username]["name"]
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password', 'error')
        else:
            flash('No registered users found', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form['fullname']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validate inputs
        if not fullname or not username or not password or not confirm_password:
            flash('Please fill in all fields', 'error')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))
        
        # Hash the password for security
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        # Check if username already exists
        users_file = "users.json"
        users = {}
        
        if os.path.exists(users_file):
            with open(users_file, "r") as f:
                users = json.load(f)
        
        if username in users:
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        
        # Add new user
        users[username] = {
            "name": fullname,
            "password": hashed_password
        }
        
        # Save to file
        with open(users_file, "w") as f:
            json.dump(users, f)
        
        flash('Registration successful! You can now login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('name', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    # Load user entries
    entries = get_user_entries(session['username'])
    
    # Calculate stats
    entry_count = len(entries)
    streak = calculate_streak(entries)
    
    # Calculate average word count
    avg_words = 0
    if entries:
        avg_words = sum(entry['word_count'] for entry in entries) // len(entries)
    
    # Get overall mood
    overall_mood = calculate_mood(entries)
    
    # Get recent entries (up to 2)
    recent_entries = sorted(entries, key=lambda x: x['date'], reverse=True)[:2]
    
    # Generate mood chart
    mood_chart = generate_mood_chart(entries)
    
    return render_template('dashboard.html', 
                          name=session['name'],
                          entry_count=entry_count,
                          streak=streak,
                          avg_words=avg_words,
                          overall_mood=overall_mood,
                          recent_entries=recent_entries,
                          mood_chart=mood_chart)

@app.route('/diary', methods=['GET', 'POST'])
def diary():
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        text = request.form['diary_text'].strip()
        entry_date = request.form['entry_date'].strip()
        language = request.form.get('language', 'auto')
        
        if not text:
            flash('Please enter some text before saving', 'error')
            return redirect(url_for('diary'))
        
        # Validate date format
        try:
            datetime.datetime.strptime(entry_date, '%Y-%m-%d')
        except ValueError:
            flash('Invalid date format. Please use YYYY-MM-DD format.', 'error')
            return redirect(url_for('diary'))
        
        # Store original text
        original_text = text
        
        # Detect language and translate if needed
        if language == 'auto':
            detected_lang, lang_name = detect_language(text)
            print(f"Auto-detected language: {detected_lang} ({lang_name})")
        else:
            detected_lang = language
            lang_names = {
                'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 
                'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 
                'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'kn': 'Kannada',
                'hi': 'Hindi', 'ar': 'Arabic', 'bn': 'Bengali', 'ta': 'Tamil', 
                'te': 'Telugu', 'ml': 'Malayalam', 'mr': 'Marathi', 'gu': 'Gujarati'
            }
            lang_name = lang_names.get(detected_lang, "Unknown")
            print(f"User selected language: {detected_lang} ({lang_name})")
        
        # Always translate if not English
        translated_text = text
        translation_failed = False
        
        if detected_lang != 'en':
            try:
                # Try translation up to 3 times
                for attempt in range(3):
                    try:
                        translated_text = translate_to_english(text, detected_lang)
                        print(f"Translation attempt {attempt+1} result: '{translated_text[:30]}...'")
                        
                        # Verify translation worked - if translated text is different from original
                        if translated_text and translated_text.strip() != original_text.strip():
                            translation_failed = False
                            print(f"Translation successful on attempt {attempt+1}")
                            break
                        else:
                            print(f"Translation attempt {attempt+1} returned same text, trying again")
                            translation_failed = True
                    except Exception as e:
                        print(f"Translation attempt {attempt+1} error: {e}")
                        translation_failed = True
                        
                        # Wait briefly before retrying
                        import time
                        time.sleep(1)
                
                if translation_failed:
                    # If all attempts failed, use original text
                    translated_text = original_text
                    flash(f"Note: Could not translate from {lang_name}. Saving original text.", "warning")
                    print("All translation attempts failed")
            except Exception as e:
                print(f"Translation error: {e}")
                translated_text = original_text
                translation_failed = True
                flash(f"Note: Could not translate from {lang_name}. Saving original text.", "warning")
        else:
            print("Text already in English, no translation needed")
        
        # Create entry object
        entry = {
            "id": int(datetime.datetime.now().timestamp()),
            "date": entry_date,
            "text": translated_text,
            "original_text": original_text if detected_lang != 'en' and not translation_failed else None,
            "original_language": lang_name if detected_lang != 'en' and not translation_failed else None,
            "word_count": len(translated_text.split()),
            "mood": analyze_mood(translated_text)
        }
        
        # Load existing entries
        entries = get_user_entries(session['username'])
        
        # Add new entry
        entries.append(entry)
        
        # Save entries
        save_user_entries(session['username'], entries)
        
        if detected_lang != 'en' and not translation_failed:
            flash(f'Entry saved successfully! (Translated from {lang_name})', 'success')
        else:
            flash('Entry saved successfully!', 'success')
        return redirect(url_for('entries'))
    
    # Get today's date in YYYY-MM-DD format for the date input
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Format date for display
    display_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
    
    return render_template('diary.html', today_date=today_date, date=display_date)

@app.route('/insights')
def insights():
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    # Load user entries
    entries = get_user_entries(session['username'])
    
    # Generate charts
    mood_chart = generate_mood_chart(entries)
    word_chart = generate_word_frequency_chart(entries)
    patterns_chart = generate_patterns_chart(entries)
    
    return render_template('insights.html', 
                          mood_chart=mood_chart,
                          word_chart=word_chart,
                          patterns_chart=patterns_chart)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'update_profile':
            fullname = request.form['fullname']
            
            if not fullname:
                flash('Please enter your full name', 'error')
                return redirect(url_for('settings'))
            
            # Update user profile
            users_file = "users.json"
            with open(users_file, "r") as f:
                users = json.load(f)
            
            users[session['username']]['name'] = fullname
            
            with open(users_file, "w") as f:
                json.dump(users, f)
            
            # Update session
            session['name'] = fullname
            
            flash('Profile updated successfully!', 'success')
            
        elif action == 'change_password':
            current_password = request.form['current_password']
            new_password = request.form['new_password']
            confirm_password = request.form['confirm_password']
            
            if not current_password or not new_password or not confirm_password:
                flash('Please fill in all password fields', 'error')
                return redirect(url_for('settings'))
            
            if new_password != confirm_password:
                flash('New passwords do not match', 'error')
                return redirect(url_for('settings'))
            
            # Hash passwords
            hashed_current = hashlib.sha256(current_password.encode()).hexdigest()
            hashed_new = hashlib.sha256(new_password.encode()).hexdigest()
            
            # Check current password
            users_file = "users.json"
            with open(users_file, "r") as f:
                users = json.load(f)
            
            if users[session['username']]['password'] != hashed_current:
                flash('Current password is incorrect', 'error')
                return redirect(url_for('settings'))
            
            # Update password
            users[session['username']]['password'] = hashed_new
            
            with open(users_file, "w") as f:
                json.dump(users, f)
            
            flash('Password changed successfully!', 'success')
            
        elif action == 'export_data':
            # Load user entries
            entries = get_user_entries(session['username'])
            
            if not entries:
                flash('No entries to export', 'error')
                return redirect(url_for('settings'))
            
            # Create export data
            export_data = {
                "user": session['name'],
                "exported_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "entries": entries
            }
            
            # Save to file
            export_file = f"export_{session['username']}_{datetime.datetime.now().strftime('%Y%m%d')}.json"
            
            with open(export_file, "w") as f:
                json.dump(export_data, f, indent=2)
            
            flash(f'Data exported successfully to {export_file}', 'success')
    
    # Get user data
    users_file = "users.json"
    with open(users_file, "r") as f:
        users = json.load(f)
    
    # Safely get user data with default values if user doesn't exist
    username = session.get('username')
    user_data = users.get(username, {})
    fullname = user_data.get('name', username) if user_data else username
    
    return render_template('settings.html', fullname=fullname, username=username)

@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text():
    # This would normally use a speech recognition API
    # For now, we'll just return a placeholder response
    return jsonify({"text": "This is a placeholder for speech recognition."})

@app.route('/entries')
def entries():
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    # Load user entries
    entries = get_user_entries(session['username'])
    
    # Sort entries by date (newest first)
    sorted_entries = sorted(entries, key=lambda x: x['date'], reverse=True)
    
    return render_template('entries.html', entries=sorted_entries)

@app.route('/entry/<int:entry_id>')
def view_entry(entry_id):
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    # Load user entries
    entries = get_user_entries(session['username'])
    
    # Find the specific entry
    entry = next((e for e in entries if e['id'] == entry_id), None)
    
    if entry is None:
        flash('Entry not found', 'error')
        return redirect(url_for('entries'))
    
    # Find previous and next entry IDs
    sorted_entries = sorted(entries, key=lambda x: x['date'], reverse=True)
    entry_ids = [e['id'] for e in sorted_entries]
    
    try:
        current_index = entry_ids.index(entry_id)
        # Previous entry is the one after current in the sorted list (newer date)
        prev_id = entry_ids[current_index + 1] if current_index < len(entry_ids) - 1 else None
        # Next entry is the one before current in the sorted list (older date)
        next_id = entry_ids[current_index - 1] if current_index > 0 else None
    except ValueError:
        prev_id = None
        next_id = None
    
    return render_template('view_entry.html', entry=entry, prev_id=prev_id, next_id=next_id)

@app.route('/entry/edit/<int:entry_id>', methods=['GET', 'POST'])
def edit_entry(entry_id):
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    # Load user entries
    entries = get_user_entries(session['username'])
    
    # Find the specific entry
    entry_index = next((i for i, e in enumerate(entries) if e['id'] == entry_id), None)
    
    if entry_index is None:
        flash('Entry not found', 'error')
        return redirect(url_for('entries'))
    
    if request.method == 'POST':
        # Print all form data for debugging
        print("Form data received:")
        for key in request.form:
            print(f"  {key}: {request.form[key]}")
        
        # Try to get the text field with different possible names
        text = None
        for field_name in ['diary_text', 'text', 'entry_text', 'content']:
            if field_name in request.form:
                text = request.form[field_name].strip()
                print(f"Found text in field: {field_name}")
                break
        
        if text is None:
            flash('Error: Text field not found in form submission', 'error')
            return redirect(url_for('edit_entry', entry_id=entry_id))
        
        new_date = request.form.get('entry_date', entries[entry_index]['date'])
        
        # Update entry
        entries[entry_index]['text'] = text
        entries[entry_index]['date'] = new_date
        entries[entry_index]['word_count'] = len(text.split())
        entries[entry_index]['mood'] = analyze_mood(text)
        
        # Save entries
        save_user_entries(session['username'], entries)
        
        flash('Entry updated successfully!', 'success')
        return redirect(url_for('view_entry', entry_id=entry_id))
    
    return render_template('edit_entry.html', entry=entries[entry_index])

@app.route('/entry/delete/<int:entry_id>', methods=['POST'])
def delete_entry(entry_id):
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    try:
        # Load user entries
        entries = get_user_entries(session['username'])
        
        # Find the entry to delete
        entry_to_delete = next((e for e in entries if e['id'] == entry_id), None)
        
        if entry_to_delete is None:
            flash('Entry not found', 'error')
            return redirect(url_for('entries'))
        
        # Remove the entry
        entries = [e for e in entries if e['id'] != entry_id]
        
        # Save updated entries
        save_user_entries(session['username'], entries)
        
        flash('Entry deleted successfully!', 'success')
        return redirect(url_for('entries'))
        
    except Exception as e:
        print(f"Error in delete_entry: {e}")
        flash('An error occurred while deleting the entry', 'error')
        return redirect(url_for('entries'))

@app.route('/entries/delete_multiple', methods=['POST'])
def delete_multiple_entries():
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    try:
        # Get entry IDs to delete
        entry_ids = request.form.getlist('entry_ids[]')
        
        if not entry_ids:
            flash('No entries selected for deletion', 'error')
            return redirect(url_for('entries'))
        
        # Convert IDs to integers
        try:
            entry_ids = [int(id) for id in entry_ids]
        except ValueError as e:
            print(f"Error converting IDs: {e}")
            flash('Error processing selected entries', 'error')
            return redirect(url_for('entries'))
        
        # Load user entries
        entries = get_user_entries(session['username'])
        
        # Count entries before deletion
        entries_count_before = len(entries)
        
        # Remove selected entries
        entries = [e for e in entries if e['id'] not in entry_ids]
        
        # Count deleted entries
        deleted_count = entries_count_before - len(entries)
        
        if deleted_count == 0:
            flash('No entries were deleted', 'warning')
            return redirect(url_for('entries'))
        
        # Save updated entries
        save_user_entries(session['username'], entries)
        
        flash(f'{deleted_count} entries deleted successfully!', 'success')
        return redirect(url_for('entries'))
        
    except Exception as e:
        print(f"Error in delete_multiple_entries: {e}")
        flash('An error occurred while deleting entries', 'error')
        return redirect(url_for('entries'))

@app.route('/generate_report', methods=['POST'])
def generate_report():
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    report_type = request.form.get('report_type', 'all')
    
    # Load user entries
    entries = get_user_entries(session['username'])
    
    if not entries:
        flash('No entries to generate report', 'error')
        return redirect(url_for('settings'))
    
    try:
        # Generate PDF report
        pdf_path = generate_pdf_report(session['name'], entries, report_type)
        
        if not pdf_path:
            flash('Failed to generate report', 'error')
            return redirect(url_for('settings'))
        
        # Generate a filename for the download
        if report_type == "monthly":
            filename = f"monthly_diary_report_{datetime.datetime.now().strftime('%Y_%m')}.pdf"
        elif report_type == "mood":
            filename = f"mood_analysis_report_{datetime.datetime.now().strftime('%Y_%m_%d')}.pdf"
        else:
            filename = f"diary_report_{datetime.datetime.now().strftime('%Y_%m_%d')}.pdf"
        
        # Send the file
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
    except Exception as e:
        print(f"Error in generate_report: {e}")
        flash('An error occurred while generating the report', 'error')
        return redirect(url_for('settings'))

@app.route('/entry/download/<int:entry_id>')
def download_entry(entry_id):
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    # Load user entries
    entries = get_user_entries(session['username'])
    
    # Find the specific entry
    entry = next((e for e in entries if e['id'] == entry_id), None)
    
    if entry is None:
        flash('Entry not found', 'error')
        return redirect(url_for('entries'))
    
    # Create a text file with the entry content
    entry_date = datetime.datetime.strptime(entry['date'], '%Y-%m-%d').strftime('%Y-%m-%d')
    filename = f"diary_entry_{entry_date}.txt"
    
    # Create the content
    content = f"Diary Entry - {entry_date}\n"
    content += f"Mood: {entry['mood']}\n"
    content += f"Word Count: {entry['word_count']}\n\n"
    content += entry['text']
    
    # Create a response with the file
    response = make_response(content)
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    response.headers["Content-Type"] = "text/plain"
    
    return response

@app.route('/translate', methods=['POST'])
def translate():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        source_lang = data.get('source_lang', 'auto')
        
        print(f"Translation request - Text: {text[:50]}..., Source lang: {source_lang}")
        
        if not text:
            return jsonify({'translation': ''})
        
        # If source language is auto, detect it
        if source_lang == 'auto':
            print("Auto-detecting language...")
            detected_lang, lang_name = detect_language(text)
            print(f"Detected language: {detected_lang} ({lang_name})")
            source_lang = detected_lang
        
        # Only translate if not English
        if source_lang != 'en':
            print(f"Translating from {source_lang} to English...")
            translated_text = translate_to_english(text, source_lang)
            print(f"Translation result: {translated_text[:50]}...")
        else:
            print("Text is already in English, no translation needed")
            translated_text = text
        
        return jsonify({
            'translation': translated_text,
            'source_lang': source_lang
        })
        
    except Exception as e:
        print(f"Translation error: {e}")
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)



























