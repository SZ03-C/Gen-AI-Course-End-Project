# PROJECT 3: TRANSFORMER-BASED SENTIMENT ANALYSIS
# Hugging Face Transformers + Gradio

!pip -q install transformers gradio

import pandas as pd
import torch
import gradio as gr
from transformers import pipeline

# 1. Load a pre-trained Transformer model
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

device = 0 if torch.cuda.is_available() else -1

sentiment_model = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    device=device
)

print("Model loaded successfully:", MODEL_NAME)

# 2. Define the prediction function
def analyze_sentiment(text):
    if not text or not text.strip():
        return {"ERROR": 1.0}

    result = sentiment_model(text[:512])[0]

    label = result["label"]
    confidence = float(result["score"])

    return {
        f"{label} ({confidence:.1%} confidence)": confidence
    }

# 3. Test with representative examples
test_sentences = [
    "This project is excellent and very useful.",
    "The application is terrible and disappointing.",
    "The presentation was good, clear, and informative.",
    "I am unhappy with the poor performance.",
    "The product arrived yesterday."
]

results = []

for sentence in test_sentences:
    prediction = sentiment_model(sentence)[0]

    results.append({
        "Input Text": sentence,
        "Predicted Sentiment": prediction["label"],
        "Confidence": round(float(prediction["score"]), 4)
    })

results_df = pd.DataFrame(results)

print("\nRepresentative Test Results:")
display(results_df)

# 4. Create the interactive application
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        lines=4,
        label="Enter Text",
        placeholder="Type a sentence to analyze its sentiment..."
    ),
    outputs=gr.Label(label="Sentiment Prediction"),
    title="Transformer-Based Sentiment Analysis",
    description=(
        "Analyze text using a pre-trained Hugging Face "
        "Transformer. The model predicts POSITIVE or NEGATIVE "
        "sentiment and provides a confidence score."
    ),
    examples=[
        ["I absolutely love this application!"],
        ["This experience was disappointing."],
        ["The explanation was clear and helpful."]
    ]
)

# 5. Launch the application
demo.launch(share=True)
