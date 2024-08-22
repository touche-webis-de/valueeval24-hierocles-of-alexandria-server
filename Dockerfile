ARG include_models=true

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS base
ENV PYTHONPATH=/
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
SHELL ["/bin/bash", "-c"]
WORKDIR /
# Install requirements
COPY requirements.txt /
RUN pip3 install -r /requirements.txt
COPY custom_models /custom_models/
RUN mkdir /models /tokenizer

FROM base AS branch-include-models-true
COPY download_model_and_tokenizer.py /
RUN python3 download_model_and_tokenizer.py

FROM base AS branch-include-models-false
# do nothing

FROM branch-include-models-${include_models} AS final
COPY predict.py /predict.py
ENTRYPOINT [ "python3", "/predict.py" ]

