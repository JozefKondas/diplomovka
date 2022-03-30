ARG BASE_IMAGE_TYPE=cpu
# these images have been pushed to Dockerhub but you can find
# each Dockerfile used in the `base_images` directory 
FROM jafermarq/jetsonfederated_$BASE_IMAGE_TYPE:latest

RUN apt-get install wget -y

# Download and extract CIFAR-10
# To keep things simple, we keep this as part of the docker image.
# If the dataset is already in your system you can mount it instead.
ENV DATA_DIR=/app/data/cifar-10
RUN mkdir -p $DATA_DIR
WORKDIR $DATA_DIR
RUN wget https://www.cs.toronto.edu/\~kriz/cifar-10-python.tar.gz 
RUN tar -zxvf cifar-10-python.tar.gz

WORKDIR /app
# Scripts needed for Flower client
#ADD client.py /app
ADD client_tf.py /app
ADD utils.py /app

########## you have to add your own dataset here
#ADD f1.csv /app

# update pip
RUN pip3 install --upgrade pip
#####################################
RUN LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

RUN apt-get update
#RUN apt-get install -y build-essential gfortan libatlas-base-dev


RUN pip3 install numpy cython sklearn

# making sure the latest version of flower is installed
RUN pip3 install flwr==0.16.0

RUN pip3 install tensorflow-aarch64 -f https://tf.kmtea.eu/whl/stable.html

RUN pip3 install keras

RUN pip3 install pandas

#################### here goes name of python script you wanna run #################
ENTRYPOINT ["python3","-u","./client_tf.py"]
