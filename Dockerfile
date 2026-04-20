FROM ollama/ollama

WORKDIR /root

COPY requirements.txt ./

RUN apt update
RUN apt-get install -y python3 python3-pip vim git
RUN pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 --break-system-packages
RUN pip install -r requirements.txt --break-system-packages

EXPOSE 8501
EXPOSE 11434
ENTRYPOINT ["./entrypoint.sh"]