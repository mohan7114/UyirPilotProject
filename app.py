import cv2
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
import sys
from ultralytics import YOLO
import cvzone
import time
import requests
from gpiozero import LED  # GPIO for Raspberry Pi 5



def show_error_window(title, message):

    """Display error message in a window"""
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showerror(title, message)
        root.destroy()

    except Exception:
        # Fallback: print if no display available
        print(f"[{title}] {message}")


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


def calculate_pedestrian_time(max_count):
    """
    Map maximum pedestrian count in the last 120s to green time (seconds).
    Adjust these rules as needed.
    """
    if max_count <= 0:
        return 0          # minimum time
    elif 1 <= max_count <= 3:
        return 10
    elif 4 <= max_count <= 5:
        return 15
    else:
        return 25         # heavy crowd


def main():
    cap = None
    vehicle_signal = None
    ped_signal = None

    try:
        # ---------------- GPIO SETUP ----------------
        try:
            # Vehicle trigger: GPIO17 (BCM) -> Physical pin 11
            # Pedestrian trigger: GPIO27 (BCM) -> Physical pin 13
            vehicle_signal = LED(17)
            ped_signal = LED(27)
            # Start with vehicle GREEN, pedestrian RED
            vehicle_signal.on()
            ped_signal.off()
            print("GPIO initialized: VEHICLE=GPIO17, PEDESTRIAN=GPIO27")
        except Exception as e:
            show_error_window(
                "GPIO Error",
                "Failed to initialize GPIO pins.\n\n"
                f"Details: {e}\n\n"
                "Check that:\n"
                "- gpiozero is installed\n"
                "- You are running on Raspberry Pi OS\n"
                "- You have permission to use /dev/gpiomem (or run with sudo)."
            )
            return

        # ---------------- YOLO MODEL ----------------
        try:
            model = YOLO('best.pt')
        except Exception as e:
            show_error_window(
                "Model Error",
                f"Failed to load YOLO model: {str(e)}\n\n"
                "Please ensure 'best.pt' exists in the current directory."
            )
            return

        # ---------------- CAMERA ----------------
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Could not access the camera")
        except Exception:
            show_error_window(
                "Camera Error",
                "Failed to access camera.\n\n"
                "Please check if:\n"
                "1. Camera is properly connected\n"
                "2. Camera is not used by another application\n"
                "3. Permissions are enabled"
            )
            return

        # ---------------- CONFIG / WINDOW ----------------
        config = AreaConfig()
        cv2.namedWindow('RGB')
        cv2.setMouseCallback('RGB', mouse_callback, config)


        print("Instructions:")
        print("- Press 's' to toggle area selection mode")
        print("- When in selection mode:")
        print("  * Left click to add points for an area")
        print("  * Right click to complete the current area")
        print("- Press 'c' to clear all custom areas")
        print("- Press 'q' to quit")


        url = "http://192.168.177.20/update"

        # ---------------- STATE MACHINE ----------------
        MAIN_GREEN_DURATION = 120  # Vehicle green duration (seconds)
        phase = "vehicle_green"    # "vehicle_green" or "ped_green"
        cycle_start_time = time.time()
        ped_phase_start_time = None
        ped_phase_duration = 0

        max_ped_count_in_cycle = 0   # Maximum pedestrians seen during 120s

        print("Starting in VEHICLE GREEN phase for 120 seconds...")

        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Failed to read frame from camera")
                 
                frame = cv2.resize(frame, (1020, 500))
                frame_height, frame_width = frame.shape[:2]
               

                # Reset per-area counters for display
                config.reset_counters()

          
                # ---------------- YOLO INFERENCE ----------------
                results = model.predict(frame)
                a = results[0].boxes.data
                px = pd.DataFrame(a).astype("float")
                total_count = 0
                area_total = 0
                cx, cy = -1, -1

                if not px.empty:
                    for index, row in px.iterrows():
                        x1, y1, x2, y2 = map(int, row[:4])
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        # Counting

                        if not config.areas:
                            total_count += 1
                        else:
                            for area_name, area_info in config.areas.items():
                                if is_inside_polygon((cx, cy), area_info['coordinates']):
                                    area_info['counter'] += 1
                                    area_total += 1

                    log_message = f"Detection at ({cx}, {cy})"
                else:
                    log_message = "No detections in this frame."

                print(log_message)

                # ---------------- DRAW UI ----------------
                mode_text = "Selection Mode: ON" if config.selecting_mode else "Selection Mode: OFF"
                cv2.putText(
                    frame, mode_text, (frame_width - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

              
                # Completed areas

                for area_name, area_info in config.areas.items():
                    cv2.polylines(
                        frame, [np.array(area_info['coordinates'], np.int32)],
                        True, area_info['color'], 2
                    )

                    label_x, label_y = area_info['coordinates'][0]

                    cv2.putText(
                        frame, area_name, (label_x, label_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, area_info['color'], 2
                    )

             
                # Current area being created
                if config.selecting_mode and len(config.current_area) > 0:
                    points = np.array(config.current_area, np.int32)
                    cv2.polylines(frame, [points], False, (255, 255, 0), 2)
                    for point in config.current_area:
                        cv2.circle(frame, point, 5, (0, 255, 0), -1)

              
                # Left side counter panel

                overlay = frame.copy()
                if config.areas:
                    panel_height = 30 + (len(config.areas) * 40)
                    cv2.rectangle(overlay, (10, 10), (200, panel_height), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                   

                    y_position = 40
                    for area_name, area_info in config.areas.items():
                        cvzone.putTextRect(
                            frame, f'{area_name}: {area_info["counter"]}',
                            (20, y_position), scale=1.5, thickness=2,
                            colorR=(255, 255, 0)
                        )
                        y_position += 40


                # Total count for this frame
                display_count = area_total if config.areas else total_count



                # ---------------- UPDATE MAX COUNT IN CYCLE ----------------

                if phase == "vehicle_green":
                    if display_count > max_ped_count_in_cycle:
                        max_ped_count_in_cycle = display_count



                # ---------------- STATE MACHINE LOGIC ----------------

                now = time.time()
                if phase == "vehicle_green":
                    elapsed = now - cycle_start_time
                    if elapsed >= MAIN_GREEN_DURATION:
                        # 120 seconds over: compute pedestrian time
                        ped_phase_duration = calculate_pedestrian_time(max_ped_count_in_cycle)
                        print(f"Vehicle green phase over. Max pedestrians = {max_ped_count_in_cycle}")
                        print(f"Pedestrian green time = {ped_phase_duration} seconds")
                        if ped_phase_duration<=0:
                            zero=1

                        # Send OFF trigger to vehicle, ON to pedestrian
                        vehicle_signal.off()
                        ped_signal.on()
                        print("TRIGGER: VEHICLE=0 (RED), PEDESTRIAN=1 (GREEN)")



                        # Optional: send count to controller server once per cycle

                        try:
                            data = {"count": max_ped_count_in_cycle}
                            response = requests.post(url, data=data)
                            print("HTTP Response:", response.text)
                        except requests.exceptions.RequestException as e:
                            print(f"HTTP request failed: {e}")

                        phase = "ped_green"
                        ped_phase_start_time = now

                elif phase == "ped_green":
                    elapsed_ped = now - ped_phase_start_time
                    if elapsed_ped >= ped_phase_duration:
                        # Pedestrian time finished Ã¢â€ â€™ back to vehicle green
                        max_ped_count_in_cycle = 0
                        cycle_start_time = now
                        phase = "vehicle_green"


                        # Trigger: vehicle green, pedestrian red
                        vehicle_signal.on()
                        ped_signal.off()
                        print("TRIGGER: VEHICLE=1 (GREEN), PEDESTRIAN=0 (RED)")
                        print("New vehicle phase started (120 seconds).")



                # ---------------- DISPLAY CURRENT PHASE ----------------

                phase_text = f"Phase: {phase}"
                cv2.putText(
                    frame, phase_text, (20, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2

                )



                # Right side total count display
                right_panel_x = frame_width - 250
                overlay = frame.copy()
                cv2.rectangle(overlay, (right_panel_x, 40), (frame_width - 20, 90), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cvzone.putTextRect(
                    frame, f'Total Count: {display_count}',
                    (right_panel_x + 10, 70), scale=1.5, thickness=2,
                    colorR=(255, 255, 0)
                )


                cv2.imshow("RGB", frame)
                key = cv2.waitKey(1) & 0xFF
               

                if key == ord('q'):

                    break

                elif key == ord('c'):

                    config.areas = {}
                    config.current_area = []

                elif key == ord('s'):

                    config.selecting_mode = not config.selecting_mode
                    if not config.selecting_mode:
                        config.current_area = []



            except Exception as e:
                show_error_window(
                    "Runtime Error",
                    f"An error occurred during execution:\n{str(e)}\n\n"
                    "The application will now close."
                )

        MAIN_GREEN_DURATION=5


    except Exception as e:

        show_error_window(
            "Fatal Error",
            f"A fatal error occurred:\n{str(e)}\n\n"
            "Please check all dependencies are properly installed."
        )

   

    finally:
        try:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass


        try:
            if vehicle_signal is not None:
                vehicle_signal.off()

            if ped_signal is not None:
                ped_signal.off()

        except Exception:

            pass



if __name__ == "__main__":

    main()


