FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# flask
EXPOSE 5000

# train model (one time)
RUN python train_pipeline.py

# start
CMD ["python", "app.py"]