import os
import sys
import time
import threading
import subprocess

# --- 1. ENVIRONMENT SETUP (CRITICAL FOR IP CAMERA) ---

# Forces OpenCV to use TCP. Crucial for stable RTSP streams on WiFi/Ethernet.

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"



import cv2

import pandas as pd

import numpy as np

import tkinter as tk

from tkinter import messagebox

from ultralytics import YOLO

import cvzone

import requests

from gpiozero import OutputDevice



# --- CONFIGURATION ---

VEHICLE_PIN = 22     # Physical Pin 15

PEDESTRIAN_PIN = 23  # Physical Pin 16

DETECTION_DURATION = 30 # Default 30 Seconds



# Camera Settings

CAMERA_IP = "192.168.29.24"

# Using Subtype=1 (SubStream) to reduce Lag. Change to 0 for MainStream if needed.

IP_CAMERA_URL = f"rtsp://admin:admin%40123@{CAMERA_IP}:554/cam/realmonitor?channel=1&subtype=1"



# Server Settings

SERVER_URL = "http://10.166.111.216:8080"



# --- THREADED CAMERA CLASS (NO LAG) ---

class ThreadedCamera:

    def __init__(self, src):

        self.src = src

        self.capture = cv2.VideoCapture(src, cv2.CAP_FFMPEG)

        # Attempt to lower internal buffer size

        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        

        self.thread = threading.Thread(target=self.update, args=())

        self.thread.daemon = True

        self.status = False

        self.frame = None

        self.stopped = False

        

        # Read one frame to verify connection

        if self.capture.isOpened():

            self.status, self.frame = self.capture.read()

        

        self.start()



    def start(self):

        self.stopped = False

        self.thread.start()



    def update(self):

        while not self.stopped:

            if self.capture.isOpened():

                # Read frames as fast as possible (draining the buffer)

                (status, frame) = self.capture.read()

                if status:

                    self.frame = frame

                    self.status = status

                else:

                    self.status = False

            time.sleep(0.005) # Tiny sleep to save CPU



    def read(self):

        # Return the latest frame immediately

        return self.status, self.frame

    

    def isOpened(self):

        return self.capture.isOpened()

    

    def release(self):

        self.stopped = True

        try:

            self.thread.join(timeout=1)

        except:

            pass

        self.capture.release()



# --- HELPER FUNCTIONS ---

def trigger_pulse(pin_number):

    """

    Relay-a 1 Second ON pannitu, udane OFF (Close) pannidum.

    """

    try:

        print(f"--- PULSE TRIGGER: Pin {pin_number} (ON for 1 sec) ---")

        relay = OutputDevice(pin_number, active_high=True, initial_value=False)

        time.sleep(1) # 1 Second Hold

        relay.close() # Release Pin

        print(f"--- PULSE END: Pin {pin_number} (OFF) ---")

    except Exception as e:

        print(f"GPIO Error: {e}")



def calculate_pedestrian_time(max_count):

    if max_count <= 0: return 10

    elif 1 <= max_count <= 3: return 15

    elif 4 <= max_count <= 5: return 20

    else: return 30



def show_error_window(title, message):

    try:

        root = tk.Tk()

        root.withdraw()

        messagebox.showerror(title, message)

        root.destroy()

    except:

        print(f"{title}: {message}")



class AreaConfig:

    def __init__(self):

        self.areas = {}

        self.current_area = []

        self.drawing = False

        self.selecting_mode = False

       

    def reset_counters(self):

        for area in self.areas.values():

            area['counter'] = 0



    def add_point(self, x, y):

        if self.selecting_mode:

            self.current_area.append((x, y))



    def complete_area(self):

        if self.selecting_mode and len(self.current_area) >= 3:

            area_name = f'Area {len(self.areas) + 1}'

            self.areas[area_name] = {

                'coordinates': self.current_area.copy(),

                'color': (0, 0, 255),

                'counter': 0

            }

        self.current_area = []



def mouse_callback(event, x, y, flags, param):

    config = param

    if event == cv2.EVENT_LBUTTONDOWN and config.selecting_mode:

        config.add_point(x, y)

    elif event == cv2.EVENT_RBUTTONDOWN and config.selecting_mode:

        config.complete_area()



def is_inside_polygon(point, polygon):

    x, y = point

    n = len(polygon)

    inside = False

    j = n - 1

    for i in range(n):

        if ((polygon[i][1] > y) != (polygon[j][1] > y) and 

            (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) /

             (polygon[j][1] - polygon[i][1]) + polygon[i][0])):

            inside = not inside

        j = i

    return inside



def main():

    try:

        # Check dependencies

        required_modules = {'cv2': cv2, 'pandas': pd, 'numpy': np, 'ultralytics': YOLO, 'cvzone': cvzone}

       

        # Initialize YOLO

        try:

            print("Loading YOLO Model...")

            model = YOLO('best.pt')

        except Exception as e:

            show_error_window("Model Error", f"Failed to load YOLO model: {str(e)}")

            return



        # Initialize Camera (THREADED)

        try:

            print(f"Connecting to IP Camera: {IP_CAMERA_URL}")

            # Replacing USB logic with Threaded IP Camera logic

            cap = ThreadedCamera(IP_CAMERA_URL)

            time.sleep(1.0) # Allow buffer to fill

            

            if not cap.isOpened(): 

                raise Exception("Could not open RTSP stream. Check network/URL.")

                

        except Exception as e:

            show_error_window("Camera Error", str(e))

            return



        config = AreaConfig()

        cv2.namedWindow('RGB')

        cv2.setMouseCallback('RGB', mouse_callback, config)



        print("System Ready.")



        # --- STATE MACHINE VARIABLES ---

        phase = "DETECTING" 

        cycle_start_time = time.time()

        pedestrian_end_time = 0

        final_ped_count = 0  # Renamed from max_ped_count_in_cycle

        

        current_detection_limit = DETECTION_DURATION 

        

        while True:    

            try:

                # Read from Threaded Camera

                ret, frame = cap.read()

                

                # Reconnection Logic

                if not ret or frame is None:

                    print(" ! Stream Lost. Reconnecting...")

                    cap.release()

                    time.sleep(2)

                    cap = ThreadedCamera(IP_CAMERA_URL)

                    time.sleep(1)

                    continue

                   

                frame = cv2.resize(frame, (1020, 500))

                frame_height, frame_width = frame.shape[:2]

                

                # ====================================================

                # PHASE 1: DETECTING (Vehicle Green)

                # ====================================================

                if phase == "DETECTING":

                    # Time Calculation

                    elapsed = time.time() - cycle_start_time

                    remaining = current_detection_limit - elapsed



                    # Draw Phase UI

                    cv2.putText(frame, f"Phase: DETECTING (Vehicle Green)", (20, 30), 

                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    cv2.putText(frame, f"Switch in: {int(remaining)}s", (20, 60), 

                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    

                    config.reset_counters()

                    results = model.predict(frame, verbose=False)

                    a = results[0].boxes.data

                    px = pd.DataFrame(a).astype("float")

                   

                    total_count = 0 

                    area_total = 0 

                   

                    if not px.empty:

                        for index, row in px.iterrows():

                            x1, y1, x2, y2 = map(int, row[:4])

                            cx = int(x1 + x2) // 2

                            cy = int(y1 + y2) // 2

                           

                            if not config.areas:

                                total_count += 1

                                w, h = x2-x1, y2-y1

                                cvzone.cornerRect(frame,(x1,y1,w,h),3,2)

                                cv2.circle(frame,(cx,cy),4,(255,0,0),-1)

                                cvzone.putTextRect(frame,f'person',(x1,y1),1,1)

                            else:

                                for area_name, area_info in config.areas.items():

                                    if is_inside_polygon((cx, cy), area_info['coordinates']):

                                        area_info['counter'] += 1

                                        area_total += 1

                                        w, h = x2-x1, y2-y1

                                        cvzone.cornerRect(frame,(x1,y1,w,h),3,2)

                                        cv2.circle(frame,(cx,cy),4,(255,0,0),-1)

                                        cvzone.putTextRect(frame,f'person',(x1,y1),1,1)

                   

                        # Logging logic (Optional)

                        # log_message = f"Detection at ({cx}, {cy})"

                        # print(log_message) 



                    display_count = area_total if config.areas else total_count

                    

                    # --- CHANGE: Always take the CURRENT count (Last Updated) ---

                    # Don't check for max, just update to current.

                    final_ped_count = display_count 

                    

                    cv2.putText(frame, f"Current Count: {final_ped_count}", (20, 90), 

                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



                    if elapsed >= current_detection_limit:

                        print(f"Detection Done. Final Count: {final_ped_count}")

                        

                        try:

                            data = {"count": final_ped_count}

                            requests.post(SERVER_URL, data=data, timeout=1) 

                            print("Data Sent to Server.")

                        except:

                            print("Server Error (Ignored)")



                        # --- LOGIC: ZERO COUNT CHECK ---

                        if final_ped_count == 0:

                            print("Zero Count: Skipping Pedestrian Phase.")

                            

                            trigger_pulse(VEHICLE_PIN)

                            

                            # Immediate Restart (No Delay)

                            final_ped_count = 0

                            cycle_start_time = time.time()

                            

                            # --- SPECIAL REQUIREMENT: 33 SECONDS ---

                            current_detection_limit = 27 # Approx 33s total loop

                            print("Vehicle Phase Restarted Immediately (Duration: 33s).")

                            

                        else:

                            # --- NORMAL FLOW (Count > 0) ---

                            trigger_pulse(PEDESTRIAN_PIN)



                            print(">>> Waiting 4 Seconds Delay... <<<")

                            time.sleep(4)



                            wait_time = calculate_pedestrian_time(final_ped_count)

                            pedestrian_end_time = time.time() + wait_time

                            phase = "ACTION"

                            print(f"Pedestrian Phase Started for {wait_time}s")



                # ====================================================

                # PHASE 2: ACTION (Pedestrian Green - Variable Time)

                # ====================================================

                elif phase == "ACTION":

                    remaining = pedestrian_end_time - time.time()

                    

                    cv2.putText(frame, "Phase: PEDESTRIAN CROSSING", (20, 30), 

                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    cv2.putText(frame, f"Wait Time: {int(remaining)}s", (20, 60), 

                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    cv2.putText(frame, "Detection Paused", (20, 90), 

                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)



                    if time.time() >= pedestrian_end_time:

                        print("Pedestrian Time Over.")

                        

                        trigger_pulse(VEHICLE_PIN)



                        print(">>> Waiting 9 Seconds Delay... <<<")

                        time.sleep(9)



                        final_ped_count = 0

                        cycle_start_time = time.time()

                        phase = "DETECTING"

                        

                        # --- RESET DURATION TO NORMAL (30s) ---

                        current_detection_limit = DETECTION_DURATION

                        print("Vehicle Detection Phase Started (Default Duration).")



                # --- COMMON UI ELEMENTS ---

                mode_text = "Selection: ON" if config.selecting_mode else "Selection: OFF"

                cv2.putText(frame, mode_text, (frame_width - 250, 30), 

                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                

                for area_name, area_info in config.areas.items():

                    cv2.polylines(frame,[np.array(area_info['coordinates'],np.int32)],

                                True, area_info['color'],2)

                    lx, ly = area_info['coordinates'][0]

                    cv2.putText(frame, area_name, (lx, ly-10), 

                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, area_info['color'], 2)

                

                if config.selecting_mode and len(config.current_area) > 0:

                    pts = np.array(config.current_area, np.int32)

                    cv2.polylines(frame, [pts], False, (255,255,0), 2)

                    for pt in config.current_area:

                        cv2.circle(frame, pt, 5, (0,255,0), -1)



                cv2.imshow("RGB", frame)

                

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'): break

                elif key == ord('c'): 

                    config.areas = {}

                    config.current_area = []

                elif key == ord('s'): 

                    config.selecting_mode = not config.selecting_mode

                    if not config.selecting_mode: config.current_area = []



            except Exception as e:

                print(f"Runtime Error: {e}")

                break



    except Exception as e:

        show_error_window("Fatal Error", str(e))

    

    finally:

        try:

            cap.release()

            cv2.destroyAllWindows()

        except: pass



if __name__ == "__main__":

    main()
