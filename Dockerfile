FROM python:3.6.15-alpine3.14

RUN apk update

RUN apk add make automake gcc g++ subversion python3-dev

WORKDIR /usr/src/app

RUN pip install pyyaml==5.1

RUN pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html

COPY . .

CMD ["python", "web_api.py"]