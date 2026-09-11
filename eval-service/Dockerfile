FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# The evaluators canonicalise document identity through this file, loaded
# from the service root. Without it load_identity_aliases() returns {} and
# correct retrievals of hash-keyed or duplicated documents score as misses.
# Regenerate with scripts/build_identity_aliases.py, then rebuild.
COPY identity_aliases.json .

EXPOSE 8005

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
