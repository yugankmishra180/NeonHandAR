# NeonHandAR

**Advanced Hand Tracking AR UI with Freehand Drawing, Shapes, Particles & Neon Glow Effects**

## Overview
NeonHandAR uses Python, OpenCV, and MediaPipe to create an interactive hand-tracking AR interface.  
- Draw freehand with your finger.  
- Pinch to cycle shapes (Circle, Square, Triangle).  
- Open hand triggers neon particles and freehand drawing.  
- Fist locks / stops drawing.  

## Demo
- Run `hand.py` and allow camera access.  
- Control gestures: Open Hand, Pinch, Fist.  

## Installation
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
