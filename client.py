import socket
import threading
import time
import cv2
import struct
from pynput import mouse
import network_utils
from gaze_tracker import GazeTracker

class ClientController:
    def __init__(self, host_ip, camera_source=0):
        self.host_ip = host_ip
        self.camera_source = camera_source
        self.mouse_controller = mouse.Controller()
        self.should_calibrate = False
        self.tracker = None
        
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
                print(f"Attempting TCP connection to {self.host_ip}:{network_utils.TCP_PORT}...")
                self.tcp_sock.connect((self.host_ip, network_utils.TCP_PORT))
                print("[TCP] Connected for clicks.")
                break
            except Exception as e:
                print(f"[TCP] Connection failed: {e}. Retrying in 2s...")
                time.sleep(2)

        # Connect TCP for gaze
        while True:
            try:
                print(f"Attempting Gaze connection to {self.host_ip}:{network_utils.GAZE_PORT}...")
                self.gaze_sock.connect((self.host_ip, network_utils.GAZE_PORT))
                print("[Gaze] Connected to send gaze data.")
                break
            except Exception as e:
                print(f"[Gaze] Connection failed: {e}. Retrying in 2s...")
                time.sleep(2)

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
                    import ctypes
                    px, py = network_utils.unpack_move(data)
                    
                    # Map percentage (0.0 - 1.0) to local screen resolution
                    user32 = ctypes.windll.user32
                    sw = user32.GetSystemMetrics(0)
                    sh = user32.GetSystemMetrics(1)
                    
                    target_x = int(px * sw)
                    target_y = int(py * sh)
                    
                    user32.SetCursorPos(target_x, target_y)
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
                elif packet_type == 4:
                    cmd_id = network_utils.unpack_control(data[:2])
                    if cmd_id == 1:
                        print("[Control] Received calibration command from Host")
                        self.should_calibrate = True
            except Exception as e:
                print(f"[TCP] Error: {e}")
                break

    def run_vision(self):
        print(f"[Vision] Initializing camera source: {self.camera_source}...")
        
        source = self.camera_source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        cap = None
        if isinstance(source, int):
            print(f"[Vision] Using local camera index {source} with DSHOW")
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            # It's a URL. Try FFMPEG first as it's usually better for network streams
            print(f"[Vision] Attempting to open URL {source} with FFMPEG...")
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                print(f"[Vision] FFMPEG failed, trying default backend...")
                cap = cv2.VideoCapture(source)

        if not cap or not cap.isOpened():
            print(f"[Vision] Error: Could not open camera source: {source}")
            print("[Vision] Troubleshooting tips:")
            print("1. Make sure DroidCam is running on your phone.")
            print(f"2. Try running: python test_connection.py {source.rsplit('/', 1)[0] if isinstance(source, str) else ''}")
            return

        self.tracker = GazeTracker()
        
        print("Waiting for Host to start calibration (Press 'C' on Host)...")
        
        last_gaze_state = None
        consecutive_failures = 0
        
        while cap.isOpened():
            if self.should_calibrate:
                print("[Vision] Starting calibration...")
                self.tracker.is_calibrated = False
                self.tracker.calibration_samples = []
                self.should_calibrate = False

            success, frame = cap.read()
            if not success:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    print("[Vision] Error: Consecutive frame grab failures. Camera might be disconnected.")
                    break
                continue
            
            consecutive_failures = 0

            annotated_image, is_looking, angles = self.tracker.process_frame(frame)
            
            if not self.tracker.is_calibrated:
                cv2.putText(annotated_image, "CALIBRATING CLIENT...", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
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
        print("Usage: python client.py <HOST_IP> [CAMERA_SOURCE]")
        print("Examples:")
        print("  python client.py 192.168.1.10")
        print("  python client.py 192.168.1.10 1")
        print("  python client.py 192.168.1.10 http://192.168.1.9:4747/video")
        sys.exit(1)
        
    host_ip = sys.argv[1]
    camera_source = sys.argv[2] if len(sys.argv) > 2 else 0
    
    client = ClientController(host_ip, camera_source)
    client.start()
