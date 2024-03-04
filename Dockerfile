FROM helen9975/deformable-detr:v0.3

WORKDIR /app
RUN pip install jupyter detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html --upgrade
RUN pip install ipython>=7.23.1 traitlets>=5.3 typing_extensions --upgrade





