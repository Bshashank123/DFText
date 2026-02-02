# DFText: AI-Assisted Handwriting Detection System

A forensic analysis tool that detects whether handwritten documents have been AI-generated or AI-edited. Built as a student project to explore the intersection of computer vision, natural language processing, and digital forensics.

## What This Does

This isn't a simple yes/no detector. It's a scoring system that tries to answer:

> *Does this handwritten image look more like something from a real pen and camera, or something AI created/edited?*

The system checks three main things:
- **Image Analysis**: How the pixels, noise, and ink behaves
- **Writing Style**: How the person writes and uses language
- **AI Language Detection**: Whether the text sounds AI-generated

## What It Can Detect

The system works with:
- Real handwritten photos taken with phones or scanners
- AI-generated handwriting (like text-to-handwriting tools)
- AI-edited handwriting where someone changed the words but kept the handwriting style

### How It Works

AI tools are pretty good at making handwriting look real, but they mess up in subtle ways:

**What AI can fake well:**
- The overall look of handwriting
- The layout of the page

**What AI struggles with:**
- Camera sensor noise patterns (every camera has a unique fingerprint)
- Natural ink behavior (real ink has randomness that's hard to copy)
- Human writing habits (we all have consistent quirks in how we write)
- Getting ALL of these perfect at the same time

The trick is looking at multiple things together - if one thing looks perfect but another looks off, that's suspicious.

## How The System Works

The basic flow is:

1. User uploads an image
2. System normalizes the image (makes it consistent for analysis)
3. OCR extracts the text
4. Three different analysis modules run at the same time:
   - Image analysis (noise, strokes, paper texture)
   - Writing style analysis (how the person writes)
   - AI text detection (does it sound like AI?)
5. All the scores get combined
6. System gives you a verdict with confidence level and explanation

No shortcuts - every image goes through all the checks.

## The Three Analysis Modules

### 1. Image Analysis

This checks if the image behaves like it came from a real camera or if it was AI-generated.

**Noise Pattern Check:**
- Real camera photos have natural noise that varies across the image
- AI images tend to have smoother, more uniform noise
- We remove the main content (blur it out) to see just the noise underneath

**Stroke Texture Check:**
- Real pen strokes have variable pressure - thicker and thinner in different spots
- AI strokes tend to be more uniform and smooth
- We measure how much the stroke width varies

**Paper Texture Check:**
- Real paper has random fibers and lighting variations
- AI-generated paper looks too clean or has subtle repeating patterns
- We look at the blank areas between text

All three scores get combined with weights:
- Noise: 40%
- Strokes: 35%
- Paper: 25%

### 2. Writing Style Analysis (Stylometry)

This is actually the strongest part of the system!

Instead of looking at what the handwriting looks like, we look at HOW the person writes:
- Average sentence length
- How much sentence length varies
- Vocabulary richness
- Word repetition patterns
- Use of connecting words (and, but, however, therefore)
- Capitalization habits
- Punctuation patterns

**The key insight:** Humans are consistent in their writing habits, even if they're not perfect. AI tends to "normalize" and "improve" writing, which shows up as unnatural changes in these patterns.

### 3. AI Text Detection

Uses a pre-trained AI text detector to check if the language sounds AI-generated. This module is weaker on its own but helps confirm the other findings.

### Final Decision

All three scores get combined:
- Writing Style: 35%
- AI Text Detection: 30%
- Image Analysis: 25%
- Other adjustments: 10%

Based on the final score:
- Under 0.40 = Likely Human
- 0.40 to 0.65 = Suspicious
- Over 0.65 = AI-Assisted Likely

The system always explains WHY it reached its conclusion.

## Tech Stack

### Backend
- **Python 3.10+** - Main programming language
- **Flask** - Web framework for the server
- **OpenCV** - Image processing
- **Pillow** - Image handling
- **pytesseract** - Text extraction from images
- **NumPy / SciPy** - Math and statistics
- **spaCy / NLTK** - Language processing
- **HuggingFace Transformers** - AI text detection

### Frontend
- HTML, CSS, Bootstrap for the web interface
- JavaScript for interactivity

Everything runs on CPU - no expensive GPU needed!

## Project Structure

```
ai_handwriting_forensics/
│
├── app.py                      # Main Flask application
│
├── pipeline/                   # Analysis modules
│   ├── preprocess.py           # Image cleanup
│   ├── ocr.py                  # Text extraction
│   ├── image_forensics.py      # Image analysis
│   ├── stylometry.py           # Writing style analysis
│   ├── ai_text.py              # AI detection
│   ├── decision.py             # Score combining
│   └── explain.py              # Results explanation
│
├── templates/                  # HTML files
│   └── index.html
│
├── static/                     # CSS, JS, images
│
└── requirements.txt            # Dependencies
```

Each module does one thing - keeps it simple and easy to understand.

## Installation

### What You Need
- Python 3.10 or higher
- Tesseract OCR
- pip

### Steps

**1. Clone the repo**
```bash
git clone https://github.com/Bshashank123/DFText.git
cd DFText
```

**2. Install Tesseract**

On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

On Mac:
```bash
brew install tesseract
```

On Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

**3. Install Python packages**
```bash
pip install -r requirements.txt
```

**4. Download language models**
```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
```

## How to Use It

### Running the App

Start the server:
```bash
python app.py
```

Then open your browser and go to `http://localhost:5000`

### Using the Web Interface
1. Go to the website
2. Upload a handwritten image (JPG, PNG, or PDF)
3. Click "Analyze Document"
4. See the results with explanations

### Example Output
```
Verdict: AI-Assisted Likely
Confidence: 76%

Reasons:
• Writing style doesn't match typical human patterns
• Language sounds AI-generated
• Image noise looks synthetic
```

## Testing It Out

Before trusting the results, you should:

1. **Test with real handwriting** - Use your own notes, letters, anything handwritten
2. **Test with AI-generated handwriting** - Use text-to-handwriting tools
3. **Look at the scores** - See if there's a clear difference between real and AI
4. **Don't expect perfection** - Some overlap is normal

### What to Expect

| Type of Input | How Well It Works |
|--------------|-------------------|
| Fully AI-generated | Easy to detect |
| Heavily edited (50%+ changed) | Moderate difficulty |
| Lightly edited (10-20% changed) | Hard to detect |
| Minimal edits (<5% changed) | Very hard |

## Limitations (Being Honest)

This isn't perfect. Here are the real limitations:

- **Single image only** - Can't compare to other samples from the same person
- **Short text is harder** - Less than 50 words doesn't give enough data
- **Minimal edits are tough** - If AI only changed a few words, it's hard to catch
- **Low quality images** - Blurry or low-res images make analysis harder
- **Not 100% certain** - This gives probabilities, not absolute proof

Anyone claiming 100% accuracy in detecting AI is lying. This tool helps you make informed decisions, but it's not a magic bullet.

## Why This Works on Gemini-Style Edits

AI tools like Gemini are good at making things look real, but they can't fake everything at once.

**What Gemini does well:**
- Makes handwriting look authentic
- Keeps the page layout looking natural
- Creates realistic-looking paper

**What Gemini can't hide:**
- Camera noise patterns (every camera leaves a unique fingerprint)
- Natural ink randomness (real ink has physics AI can't perfectly copy)
- Human writing habits (we all write consistently, even if imperfectly)
- Doing ALL of these perfectly at the same time

The system looks for inconsistencies across all these areas. One perfect fake area + other suspicious areas = probably AI.

## If You Want to Build This Yourself

If you're building from scratch, do it in this order (each step gives you a working prototype):

1. Basic image upload and display
2. OCR text extraction
3. AI text detector
4. Writing style analysis
5. Combine scores (text-only version works here!)
6. Add noise analysis
7. Add stroke analysis
8. Add paper analysis
9. Full system with explanations

You can stop at step 5 and still have something useful.

## Future Plans

Things I'd like to add:
- Batch processing (analyze multiple images at once)
- Comparison mode (compare suspicious doc to known real samples)
- Export reports as PDFs
- Support for more languages
- Better calibration with more test data
- Docker support for easier deployment

## Contributing

Want to help improve this? Great! Here's how:

1. Fork the repo
2. Make your changes
3. Test them
4. Submit a pull request

Areas that need help:
- Better analysis techniques
- Support for other languages
- Performance improvements
- More test data
- Better documentation

## License

MIT License - use it however you want!

## Contact

Made by: [Bshashank123](https://github.com/Bshashank123)

Repo: [https://github.com/Bshashank123/DFText](https://github.com/Bshashank123/DFText)

Found a bug or have a question? [Open an issue](https://github.com/Bshashank123/DFText/issues)

---

## The Main Idea

Remember: This doesn't detect "fake handwriting."

It detects **violations of human consistency** across pixels, noise, and language.

That's what makes it work - looking at multiple things that AI struggles to fake perfectly at the same time.

The goal isn't to be 100% perfect - it's to give you useful information to make better decisions.
