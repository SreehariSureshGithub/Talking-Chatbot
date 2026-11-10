# Talking Chatbot: intent-based voice chatbot

A conversational chatbot that understands typed messages, picks a reply, and **speaks it aloud**. It uses a small neural-network intent classifier (TensorFlow/Keras) with NLTK preprocessing, a Tkinter chat window, and offline text-to-speech with `pyttsx3`.

## Highlights

- **Intent classification**: a 3-layer feed-forward network (128 → 64 → softmax, with dropout) trained on bag-of-words features across **26 intents** (greetings, small talk, hobbies, movies, music, food, weather, technology, compliments and more).
- **NLP pipeline**: NLTK tokenisation and WordNet lemmatisation, plus **WordNet synonym expansion** so phrasings the model never saw still map to the right intent.
- **Confidence threshold** (0.25): low-confidence predictions are filtered out instead of returning a random answer.
- **Voice output**: every response is spoken through `pyttsx3`, which works offline.
- **Desktop GUI**: a Tkinter chat window with a scrolling history, a Send button and Enter-to-send.

## Tech stack

Python · TensorFlow / Keras · NLTK (WordNet) · NumPy · pyttsx3 · Tkinter

## How it works

```
user message ──► tokenize + lemmatize ──► synonym expansion ──► bag-of-words vector
                                                                     │
                     spoken + shown in GUI ◄── random response ◄── intent (softmax > 0.25)
```

1. `intents.json` defines each intent's tag, example patterns and possible responses.
2. `chatbot.ipynb` builds the vocabulary, turns each pattern into a bag-of-words vector, and trains the network (SGD with Nesterov momentum, 200 epochs). It saves `chatbot_model.h5`, `words.pkl` and `classes.pkl`.
3. `main.ipynb` loads the model, classifies each message, picks a response for the predicted intent, shows it in the chat window and reads it aloud.

## Project structure

```
Talking-Chatbot/
├── intents.json        # intents: tags, patterns, responses
├── Dataset.txt         # copy of the intent dataset
├── chatbot.ipynb       # preprocessing + model training
├── main.ipynb          # Tkinter GUI + inference + text-to-speech
├── chatbot_model.h5    # trained Keras model
├── words.pkl / classes.pkl / tokenizer.pickle  # saved vocabulary and labels
```

## Run it

```bash
pip install tensorflow nltk numpy pyttsx3 pyspellchecker
python -m nltk.downloader punkt wordnet
```

Open `chatbot.ipynb` and run all cells to retrain (optional; a trained model is included). Then run `main.ipynb` to start the chat window.

## Possible improvements

- Replace bag-of-words with sentence embeddings or a small transformer for better generalisation.
- Add speech-to-text input for full voice conversations.
- Package as a `.py` app or web UI and add a fallback LLM for out-of-scope questions.
