FROM kaakaa/opencv-contrib-python3

RUN pip install requests bottle==0.12.8 python-dotenv

WORKDIR /usr/local/src
