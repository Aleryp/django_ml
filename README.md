# Django ML Project

## RUN PROJECT

To run this project You just needed to have Docker and Docker Compose Installed on Your computer.

Then, from home dirrectory just run from project directory:
```sh
docker-compose up
```


## TRAIN MODELS

To train models You need to have TensorFlow installed in Your virtual environment.
Once it done, just run this from project directory:

For clean training:
```sh
python custom_training.py
```

For training on MobileNetV2:
```sh
python train_on_base.py
```
