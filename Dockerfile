FROM python:3.10.2
EXPOSE 8501
CMD mkdir -p /app
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY . .
ENTRYPOINT [ "streamlit", "run" ]
CMD ["application.py"]