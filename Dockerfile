# Verwenden Sie eine offizielle Python-Laufzeit als Basis-Image
FROM python:3-slim
 
FROM pytorch/pytorch
WORKDIR /app

# Installieren Sie git
RUN apt-get update && apt-get install -y git
RUN apt update && apt install -y libgl1 libglib2.0-0


# Ändern Sie das Arbeitsverzeichnis in das YOLOv5-Verzeichnis
WORKDIR /app/yolo
COPY . /app/yolo

# Installieren Sie die erforderlichen Pakete
RUN pip install torch torchvision torchaudio

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

