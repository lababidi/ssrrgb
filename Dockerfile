FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

###############################################################################
#   Here are three example Dockerfile.
#
#   Each performer will build a container that contains
#   all of the performer's dependencies and executes
#   the submitted model(s). Please read included README.pdf
#   for details on building and running this example container.
#
#   Absolute Requirements:
#   1.  The line 'FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04'
#       should not be altered or removed if at all possible.
#       CUDA 8.0 is a hard requirement but cuDNN can be changed at
#       their own risk. The contest organizers have verified this
#       specific version with their hardware.
#   2.  Cuda 8.0 + cuDNN 6.0 is required.
#   3.  The final command 'ENTRYPOINT' must behave identical (I/O)
#       to the example '/performer/run.py' script. Note, it does
#       not have to be Python.
#   4.  All dependencies, scripts, and local packages must
#       be added to the container. If you cannot pull from Dockerhub
#       on a fresh computer and run it with no additional effort,
#       then the container has not been configured correctly.
###############################################################################

# Here are some base libraries and apps that are installed.
# Feel free to add and remove any as needed.
RUN apt-get update -y && apt-get install -y \
  cmake git wget ca-certificates build-essential \
    libssl-dev libffi-dev

# Here are some base python libraries that are installed.
# Feel free to add and remove any as needed.
RUN apt-get install -y python3-dev \
  python3-pip libopencv-dev \
      python3-numpy python3-scipy python3-matplotlib

RUN pip3 --no-cache-dir install \
        opencv-python \
        Keras==2.0.8 \
        Pillow \
        scikit-learn==0.19.0 \
        scikit-image==0.13.1 \
        h5py


# This is just an example of copying the performer's
# model(s) into the docker container. When docker build
# is executed the this will remain in the build and
# on the repo that is pulled for evaluation. This
# can be changed to whatever the performer wants
COPY performer /root/performer

# RUN, COPY, or do anything performer needs before
# this entrypoint.

# This is an example entrypoint for the script.
# Performer must have their script automatically
# run and perform in the same manner as the
# provided example run.py. However, python does
# not have to be used as long as the outcome is
# identical.
ENTRYPOINT ["python3", "/root/performer/run.py"]


