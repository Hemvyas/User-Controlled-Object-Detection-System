API_KEY = "kjqQE29cDa4SqSZO0uyU"
API_URL = "https://detect.roboflow.com"

QUERY_TO_MODEL_ID = {
    "student": ("class-room-management-system/1", ["students"]),
    "look around up down": ("cctv-accurate/1", ["lookaround","up","down"]),
    "notebook empty chair": ("trab-2-wmjig/1",["notebook","empty chair"]),
    "tables": ("classroom-table-detection/1",["tables"]),
    "humans": ("crowd-count-rcr1e/1",["humans"]),
    # "distracted": ("student-attention-tracking/1"),
    "distracted attentive sleepy": ("student-attention-tracking/1",["distracted","attentive","sleepy"]),
    "whiteboard": ("whiteboard-detection/1",["whiteboard"]),
    "wearing id": ("id-bzx4h/5",["wearing id card"]),
    "rase hands": ("handraising-detection-project-v.2/5",["rase hand"]),
    "phone": ("detector-de-celulares-t9zxc/1",["phone","mobile"]),
}
