FROM python:3.9-slim

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN apt-get update
RUN apt-get install neovim tmux\
	python3-opencv \
	curl \
	bash bash-completion -y
RUN chsh -s $(which bash)

COPY requirements.txt .
RUN cat requirements.txt | grep -v torch | tee requirements.txt
RUN pip install -r requirements.txt

RUN mkdir /root/working
WORKDIR /root/working
COPY . .

CMD ["streamlit", "run", "main_doctr.py"]
