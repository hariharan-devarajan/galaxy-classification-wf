FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN mkdir app
WORKDIR /app

RUN python3 -m pip install --upgrade pip
RUN pip3 install \
    opencv-python \
    optuna==2.0.0 \
    pandas \
    matplotlib \
    torch \
    numpy \
    Pillow \
    bs4 \
    scikit-learn \
    torchvision \
    pytorchtools \
    joblib\
    scikit-image \
    pathlib \
    seaborn \
    scikit-plot \
    torch-summary

RUN mkdir ./bin
COPY preprocess_resize.py ./bin
RUN chmod 777 ./bin/preprocess_resize.py
COPY preprocess_augment.py ./bin
RUN chmod 777 ./bin/preprocess_augment.py
COPY vgg16_hpo.py ./bin
RUN chmod 777 ./bin/vgg16_hpo.py
COPY data_loader.py ./bin
RUN chmod 777 ./bin/data_loader.py
COPY model_selection.py ./bin
RUN chmod 777 ./bin/model_selection.py
COPY train_model.py ./bin
RUN chmod 777 ./bin/train_model.py
COPY eval_model.py ./bin
RUN chmod 777 ./bin/eval_model.py
