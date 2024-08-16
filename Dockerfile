FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONPATH=/

RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

SHELL ["/bin/bash", "-c"]
WORKDIR /

# Install requirements
COPY requirements.txt /
RUN pip3 install -r /requirements.txt

COPY custom_models /custom_models/

# pre-download tokenizer and model
RUN mkdir /models
RUN mkdir /tokenizer
COPY download_model_and_tokenizer.py /
RUN python3 download_model_and_tokenizer.py

# copy script files
COPY predict.py /predict.py

ENTRYPOINT [ "python3", "/predict.py" ]
