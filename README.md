# ui-service

Gradio front door: Chat, Documents, Dashboard tabs.
Ships in MOCK_MODE=true by default so it runs standalone before
orchestrator-api exists.

## Run

```
pip install -r requirements.txt
python app.py
```

## Switch to real orchestrator

```
ORCHESTRATOR_URL=http://localhost:8000
MOCK_MODE=false
```
