# Transformer-Based Sentiment Analysis

## Objective
Build an interactive sentiment-analysis application using a pretrained Hugging Face Transformer.

## Model
`distilbert/distilbert-base-uncased-finetuned-sst-2-english`, a DistilBERT model fine-tuned for binary sentiment classification (POSITIVE / NEGATIVE).

## Application
A Gradio interface accepts text and displays the predicted label and confidence score. The notebook also tests five representative sentences.

## Run
```bash
pip install -r requirements.txt
python app.py
```
Internet access is needed to download the model on first run. The Gradio share URL is temporary when launched with `share=True`.

## Evaluation note
The five example sentences are functional tests, not a formal benchmark. Neutral text may still be classified as positive or negative because the model is binary.

## Files
`app.py` contains the standalone implementation; `sentiment_analysis.ipynb` preserves the executed notebook and outputs.
