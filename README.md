# Face Recognizer

from tutorial : https://realpython.com/face-recognition-with-python/ <br>
with dataset: https://www.kaggle.com/datasets/adg1822/7-celebrity-images

## Image recognition

```sh
❯ python img_recognizer.py
usage: img_recognizer.py [-h] [--train] [--validate] [--test] [-m {hog,cnn}] [-f F]

Image face recognition

options:
  -h, --help    show this help message and exit
  --train       Train the model
  --validate    Validate trained model
  --test        Test the model with an unknown image
  -m {hog,cnn}  Which model to use for training: hog (CPU), cnn (GPU)
  -f F          Path to an image with an unknown face
```

## Video recognition

```sh
❯ python video_recognizer.py
usage: video_recognizer.py [-h] [-f F] [-d] [-o OUTPUT]

Video face recognition

options:
  -h, --help            show this help message and exit
  -f F                  Video file path
  -d, --display         Display video
  -o OUTPUT, --output OUTPUT
                        Output file path
```

Sample:

```sh
python video_recognizer.py -f path/to/sample.mp4 -o path/to/result.mp4 --display
```
