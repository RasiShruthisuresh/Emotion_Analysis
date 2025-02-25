# Emotion Analysis Dashboard

A Streamlit application that analyzes customer feedback and provides emotion analysis with visualizations.

## Features

- Text feedback input
- Emotion analysis using transformer-based model
- Interactive radar chart visualization
- Emotion scores in JSON format
- Adore score calculation
- Theme analysis display

## Setup

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Run the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Usage

1. Enter your feedback text in the text area
2. Click "Analyze" button
3. View the results:
   - Radar chart showing emotion distribution
   - Adore score
   - Detailed emotion analysis in JSON format
   - Top themes in the dataset
