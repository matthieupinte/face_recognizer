from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw

import argparse
import face_recognition
import pickle

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = (0, 0, 255) # Blue
TEXT_COLOR = (255, 255, 255) # White

Path("training").mkdir(parents=True, exist_ok=True)
Path("output").mkdir(parents=True, exist_ok=True)
Path("validation").mkdir(parents=True, exist_ok=True)

def _display_face(draw, bounding_box, name):
  top, right, bottom, left = bounding_box
  draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
  text_left, text_top, text_right, text_bottom = draw.textbbox(
    (left, bottom - 30), name
  )
  draw.rectangle(
    ((text_left, text_top), (text_right, text_bottom)), fill=BOUNDING_BOX_COLOR, outline=BOUNDING_BOX_COLOR
  )
  draw.text((left + 6, bottom - 30), name, fill=TEXT_COLOR)

def encode_known_faces(
  model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
  names = []
  encodings = []
  for filepath in Path("training").rglob("*.jpg"):
    name = filepath.parent.name
    image = face_recognition.load_image_file(filepath)

    face_locations = face_recognition.face_locations(image, model=model)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for encoding in face_encodings:
      names.append(name)
      encodings.append(encoding)

  name_encodings = { "names": names, "encodings": encodings }
  with encodings_location.open(mode="wb") as f:
    pickle.dump(name_encodings, f)

def _recognize_face(unknown_encoding, loaded_encodings):
  boolean_matches = face_recognition.compare_faces(
    loaded_encodings["encodings"], unknown_encoding
  )
  votes = Counter(
    name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match
  )
  if votes:
    return votes.most_common(1)[0][0]

def recognize_faces(
  input_image: [] = None,
  model: str = "hog",
  encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
  with encodings_location.open(mode="rb") as f:
    loaded_encodings = pickle.load(f)

  faces = []

  input_face_locations = face_recognition.face_locations(input_image, model=model)
  input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

  for bounding_box, unknown_encoding in zip(
    input_face_locations, input_face_encodings
  ):
    name = _recognize_face(unknown_encoding, loaded_encodings)
    if not name:
      name = "unknown"
    faces.append([name, bounding_box])

  return faces

def validate(model: str = "hog"):
  for filepath in Path("validation").rglob("*.jpg"):
    if filepath.is_file():
      image = load_image(str(filepath.absolute()))

      faces = recognize_faces(input_image=image, model=model)

      for name, (top, right, bottom, left) in faces:
        print(f"Found {name} in {filepath.parent.joinpath(filepath.name)}")

def load_image(image_location: str):
  return face_recognition.load_image_file(image_location)

def test(image_location: str, model: str = "hog"):
  image = load_image(image_location)

  faces = recognize_faces(input_image=image, model=model)

  pillow_image = Image.fromarray(image)
  draw = ImageDraw.Draw(pillow_image)

  for name, (top, right, bottom, left) in faces:
    _display_face(draw, (top, right, bottom, left), name)

  del draw
  pillow_image.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Image face recognition")
  parser.add_argument("--train", action="store_true", help="Train the model")
  parser.add_argument("--validate", action="store_true", help="Validate trained model")
  parser.add_argument("--test", action="store_true", help="Test the model with an unknown image")
  parser.add_argument(
      "-m",
      action="store",
      default="hog",
      choices=["hog", "cnn"],
      help="Which model to use for training: hog (CPU), cnn (GPU)",
  )
  parser.add_argument(
      "-f", action="store", help="Path to an image with an unknown face"
  )
  args = parser.parse_args()

  if args.train:
    encode_known_faces(model=args.m)
  elif args.validate:
    validate(model=args.m)
  elif args.test:
    test(image_location=args.f, model=args.m)
  else:
    parser.print_help()
