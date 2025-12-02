
This project aims to detect speed limit signs on roads (specifically the non north-america
ones with the red circle around them). Although intended to recognize the value of the speed
limit as well, this part has shown to be a little unreliable. It should be noted that even
though it often reads the wrong value, it is *very* good at detecting when there is and is
not a speed limit sign in an image even if it thinks its the wrong speed.

This reposititory contains the trained model. (or can be retrained with `./train.sh`).

Create virtual environment and install requirements from `requirements.txt`.

```sh
python3 -m venv venv
source ./venv/bin/activate
python3 install -r requirements.txt
```

To calculate the graph in `confusion.png` run:

```sh
python3 demo.py
```

If you have a video you want to test, put it in line 11 of `demo2.py` and run:

```sh
python3 demo2.py
```

`convert.py` and `split.py` were use to translate the model to YOLO and split the GTSDB dataset.
