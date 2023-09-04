#!/bin/bash


sudo apt-get update

# install docker
sudo apt-get install docker.io

# start docker service
sudo systemclt start docker
sudo systemclt enable docker

# install docker-compose
sudo curl -L https://github.com/docker/compose/releases/download/1.29.2/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# build and run docker container
cd ~/server
sudo docker-compose up --build