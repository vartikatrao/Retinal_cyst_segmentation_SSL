FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .  
RUN pip --timeout=1000 install -r requirements.txt

COPY app.py . 
COPY models models 
COPY Templates Templates 

EXPOSE 5000

CMD ["python", "app.py"]

