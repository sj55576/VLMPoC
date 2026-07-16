from app.vision.geometry import calculate_joint_angle, distance, inside_region, iou, normalized_distance
from app.vision.models import Keypoint

def test_iou_and_distance():
    assert iou((0,0,2,2),(1,1,3,3)) == 1/7
    assert distance((0,0),(3,4)) == 5
    assert normalized_distance((0,0),(100,100),100,100) == 2**.5
    assert iou((3, 3, 1, 1), (0, 0, 2, 2)) == 1 / 7
    assert iou((0, 0, 0, 2), (0, 0, 2, 2)) == 0

def test_angle_and_region():
    assert round(calculate_joint_angle(Keypoint(x=1,y=0,confidence=1),Keypoint(x=0,y=0,confidence=1),Keypoint(x=0,y=1,confidence=1))) == 90
    assert inside_region((.8,.8),{"x1":.7,"y1":.7,"x2":.9,"y2":.9})
