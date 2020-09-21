FROM leesangha/fairface as build

FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y \
    python3.7 \
    python3-pip 

WORKDIR /app
COPY requirements.txt /app


RUN pip3 install --upgrade pip setuptools
RUN pip3 install cmake
RUN pip3 install dlib
RUN pip3 install --no-cache-dir torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -r requirements.txt

RUN mkdir /app/fair_face_models
COPY --from=build /app/fair_face_models /app/fair_face_models


COPY . /app

EXPOSE 80
CMD python3 app.py