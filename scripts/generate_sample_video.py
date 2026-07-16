"""Generate a dependency-light synthetic SOP video using OpenCV when installed."""
from pathlib import Path

def main() -> None:
    try: import cv2, numpy as np
    except ImportError as exc: raise SystemExit("Install project dependencies first: pip install -e .") from exc
    target = Path("data/sample_assembly.mp4"); target.parent.mkdir(exist_ok=True)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640,480))
    for i in range(60):
        frame=np.zeros((480,640,3),np.uint8); cv2.rectangle(frame,(90,35),(330,450),(255,180,0),2); cv2.putText(frame,"person + helmet",(100,30),0,.7,(255,255,255),1)
        if i>=8: cv2.rectangle(frame,(215,268),(242,335),(0,255,255),-1); cv2.putText(frame,"screwdriver",(200,360),0,.5,(255,255,255),1)
        if i>=14: cv2.rectangle(frame,(500,350,585,425) if i>=45 else (245,285,335,370),(0,180,0),-1); cv2.putText(frame,"part_a",(450 if i>=45 else 245,280),0,.5,(255,255,255),1)
        cv2.putText(frame,f"frame {i}: SOP mock timeline",(20,465),0,.6,(255,255,255),1); writer.write(frame)
    writer.release(); print(target)
if __name__ == "__main__": main()
