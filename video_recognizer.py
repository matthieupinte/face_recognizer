import argparse
import cv2
import numpy as np

from img_recognizer import recognize_faces
from progress_bar import print_progress_bar

def process_video(file_path: str, output_path: str = None, display: bool = False):
  print("Processing video...")

  cap = cv2.VideoCapture(file_path)
  fps = cap.get(cv2.CAP_PROP_FPS)
  output = None

  if (cap.isOpened() == False):
    print("Error opening video stream or file")

  i = 0
  nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  print_progress_bar(i, nb_frames, prefix="Progress:", suffix="Complete", length=50)
  while(cap.isOpened()):
    print_progress_bar(i, nb_frames, prefix="Progress:", suffix="Complete", length=50)

    ret, frame = cap.read()

    if not ret: break

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    faces = recognize_faces(input_image=rgb_frame)

    for name, (top, right, bottom, left) in faces:
      top *= 4; right *= 4; bottom *= 4; left *= 4

      cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
      cv2.putText(frame, name, (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    if display is True:
      cv2.imshow('Live', frame)

    if output is None and output_path is not None:
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      output = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]), True)

    if output is not None:
      output.write(frame)

    i += 1

  cap.release()
  cv2.destroyAllWindows()

  if output is not None:
    output.release()

  print("\rCompleted !", end="")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Video face recognition")
  parser.add_argument("-f", type=str, help="Video file path")
  parser.add_argument("-d", "--display", action="store_true", help="Display video")
  parser.add_argument("-o", "--output", type=str, help="Output file path")

  args = parser.parse_args()

  if args.f:
    process_video(file_path=args.f, output_path=args.output, display=args.display)
  else:
    parser.print_help()
