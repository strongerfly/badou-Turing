FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
ADD . /app
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
RUN echo "conda activate bilinear" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update


# The code to run when container is started:
RUN ["chmod", "+x", "/app/entrypoint.sh"]
ENTRYPOINT ["/app/entrypoint.sh"]