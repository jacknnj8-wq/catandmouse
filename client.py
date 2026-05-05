import socket
import threading
import time
import cv2
import struct
from pynput import mouse
import network_utils
from gaze_tracker import GazeTracker

class ClientController:
    def __init__(self, host_ip):
        self.host_ip = host_ip
        self.mouse_controller = mouse.Controller()
        
        # Sockets
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_sock.bind(('0.0.0.0', network_utils.UDP_PORT))
        
        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        self.gaze_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.gaze_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def start(self):
        print(f"Connecting to Host {self.host_ip}...")
        
        # Connect TCP for clicks
        while True:
            try:
                self.tcp_sock.connect((self.host_ip, network_utils.TCP_PORT))
                print("[TCP] Connected for clicks.")
                break
            except Exception as e:
                print("Retrying TCP connection...")
                time.sleep(1)

        # Connect TCP for gaze
        while True:
            try:
                self.gaze_sock.connect((self.host_ip, network_utils.GAZE_PORT))
                print("[Gaze] Connected to send gaze data.")
                break
            except Exception as e:
                print("Retrying Gaze connection...")
                time.sleep(1)

        # Start listening threads
        threading.Thread(target=self.listen_udp, daemon=True).start()
        threading.Thread(target=self.listen_tcp, daemon=True).start()
        
        # Start Vision processing in main thread
        self.run_vision()

    def listen_udp(self):
        print("[UDP] Listening for mouse movement...")
        while True:
            try:
                data, addr = self.udp_sock.recvfrom(1024)
                if not data:
                    continue
                # The first byte is the packet type
                packet_type = struct.unpack('!B', data[:1])[0]
                if packet_type == 1:
                    dx, dy = network_utils.unpack_move(data)
                    self.mouse_controller.move(dx, dy)
            except Exception as e:
                print(f"[UDP] Error: {e}")

    def listen_tcp(self):
        print("[TCP] Listening for clicks/scrolls...")
        while True:
            try:
                # We expect small packets, max 9 bytes for scroll
                data = self.tcp_sock.recv(9)
                if not data:
                    break
                
                packet_type = struct.unpack('!B', data[:1])[0]
                
                if packet_type == 2:
                    button_id, pressed = network_utils.unpack_click(data[:3])
                    
                    if button_id == 1:
                        btn = mouse.Button.left
                    elif button_id == 2:
                        btn = mouse.Button.right
                    else:
                        btn = mouse.Button.middle
                        
                    if pressed:
                        self.mouse_controller.press(btn)
                    else:
                        self.mouse_controller.release(btn)
                elif packet_type == 3:
                    dx, dy = network_utils.unpack_scroll(data[:9])
                    self.mouse_controller.scroll(dx, dy)
            except Exception as e:
                print(f"[TCP] Error: {e}")
                break

    def run_vision(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Vision] Error: Could not open camera.")
            return

        tracker = GazeTracker()
        
        print("Please look directly at the camera for calibration.")
        
        last_gaze_state = None
        consecutive_failures = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    print("[Vision] Error: Consecutive frame grab failures. Camera might be disconnected.")
                    break
                continue
            
            consecutive_failures = 0

            annotated_image, is_looking, angles = tracker.process_frame(frame)
            
            # Send state if it changed, or periodically
            if is_looking != last_gaze_state:
                try:
                    data = network_utils.pack_gaze(is_looking)
                    self.gaze_sock.sendall(data)
                    last_gaze_state = is_looking
                except Exception as e:
                    print(f"Failed to send gaze state: {e}")
            
            try:
                cv2.imshow('Client Gaze Tracker', annotated_image)
            except Exception as e:
                print(f"[Vision] Warning: Could not display window: {e}")
                
            if cv2.waitKey(5) & 0xFF == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python client.py <HOST_IP>")
        sys.exit(1)
        
    client = ClientController(sys.argv[1])
    client.start()
