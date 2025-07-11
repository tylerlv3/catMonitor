import random
import time

activity_list = ['eating', 'drinking', 'sleeping', 'playing', 'grooming', 'crying', 'other']
last_activity = {"activity": None, "time": 0}
ACTIVITY_DELAY = 5

def classify(frame):
    current_time = time.time()
    if last_activity["activity"] is None or current_time - last_activity["time"] > ACTIVITY_DELAY:
        last_activity["activity"] = random.choice(activity_list)
        last_activity["time"] = current_time
    return last_activity["activity"]
