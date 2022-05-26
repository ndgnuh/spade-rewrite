FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

COPY requirements.txt .
RUN cat requirements.txt | grep -v torch | tee requirements.txt
RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get install neovim tmux\
	python3-opencv \
	curl \
	bash bash-completion -y
RUN chsh -s $(which bash)

ENTRYPOINT ["bash"]
CMD ["read"]
