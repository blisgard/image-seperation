# People detection and adding meta-data

## Preview of files:

### Feyza-ornekler

Contains example images to use in this repo.

### yolov3

Much simpler version of yolov3 , implemented for people detection.

## Main Idea:

1 - People detection 

2 - Update metadata of image 

3 - If image contains image, run emotion detection

4 - Find smiling people and update metadata of image

## How to run:

`
$ git clone https://github.com/blisgard/image-seperation`

`$ pip install -r requirements.txt`

` $ cd yolov3 &&
python3 detect.py`

## Notes:

- You need to have ExifTool executable to use this repo.
- In /yolov3, there exists a file called **exif.config** where you can add your user-defined image-tags. 


