import socket
import threading
import time
import cv2
from pynput import mouse
import network_utils
from gaze_tracker import GazeTracker

class HostController:
    def __init__(self, camera_source=0):
        self.camera_source = camera_source
        self.active_client_ip = None
        self.gaze_states = {"host": False} # ip -> bool
        
        # Sockets
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # TCP Server for Clicks/Commands
        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp_server.bind(('0.0.0.0', network_utils.TCP_PORT))
        self.tcp_server.listen(5)
        
        # TCP Server for Gaze State
        self.gaze_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.gaze_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.gaze_server.bind(('0.0.0.0', network_utils.GAZE_PORT))
        self.gaze_server.listen(5)
+
        self.client_tcp_sockets = {} # IP -> socket

        # Mouse tracking for deltas
        self.last_pos = None
        self.mouse_controller = mouse.Controller()
        
        # Calibration state
        self.is_calibrating = False

    def start(self):
        print(f"Starting Host Server (Camera: {self.camera_source})...")
        threading.Thread(target=self.accept_tcp_clients, daemon=True).start()
        threading.Thread(target=self.accept_gaze_clients, daemon=True).start()
        
        # Start Vision processing in a separate thread
        threading.Thread(target=self.run_vision, daemon=True).start()
        
        # Start Mouse Listener
        with mouse.Listener(
                on_move=self.on_move,
                on_click=self.on_click,
                on_scroll=self.on_scroll) as listener:
            listener.join()

    def accept_tcp_clients(self):
        while True:
            client, addr = self.tcp_server.accept()
            print(f"[TCP] Client connected: {addr[0]}")
            self.client_tcp_sockets[addr[0]] = client

    def accept_gaze_clients(self):
        while True:
            client, addr = self.gaze_server.accept()
            print(f"[Gaze] Client connected: {addr[0]}")
            threading.Thread(target=self.handle_gaze_client, args=(client, addr[0]), daemon=True).start()

    def handle_gaze_client(self, client, ip):
        try:
            while True:
                data = client.recv(1)
                if not data:
                    break
                is_looking = network_utils.unpack_gaze(data)
                self.gaze_states[ip] = is_looking
                self.update_focus()
        except Exception as e:
            print(f"[Gaze] Error with {ip}: {e}")
        finally:
            client.close()
            if ip in self.gaze_states:
                del self.gaze_states[ip]
            self.update_focus()

    def update_focus(self):
        # Priority 1: Host
        if self.gaze_states.get("host"):
            if self.active_client_ip is not None:
                print("[*] Focus returned to Host")
                self.active_client_ip = None
            return

        # Priority 2: Clients
        for ip, is_looking in self.gaze_states.items():
            if ip == "host": continue
            if is_looking:
                if self.active_client_ip != ip:
                    print(f"[*] Focus switched to client {ip}")
                    self.active_client_ip = ip
                return
        
        # If no one is looking, keep current or clear?
        # Let's clear focus if no one is looking for simplicity
        # if self.active_client_ip:
        #    print("[*] No focus detected")
        #    self.active_client_ip = None

    def run_vision(self):
        print(f"[Vision] Initializing camera source: {self.camera_source}...")
        
        source = self.camera_source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        cap = None
        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source)

        if not cap or not cap.isOpened():
            print(f"[Vision] Error: Could not open camera source: {source}")
            return

        tracker = GazeTracker()
        print("Press 'C' to start coordinated calibration.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            annotated_image, is_looking, angles = tracker.process_frame(frame)
            self.gaze_states["host"] = is_looking
            self.update_focus()
            
            # Display info
            text = "LOOKING AT HOST" if is_looking else "HOST NOT FOCUSED"
            color = (0, 255, 0) if is_looking else (0, 0, 255)
            cv2.putText(annotated_image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(annotated_image, "Press 'C' to Calibrate", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            if self.is_calibrating:
                cv2.putText(annotated_image, "CALIBRATING HOST...", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if tracker.is_calibrated:
                    self.is_calibrating = False
                    print("[Vision] Host calibration complete. Triggering Clients...")
                    self.broadcast_calibration()

            cv2.imshow('Host Gaze Tracker', annotated_image)
            
            key = cv2.waitKey(5) & 0xFF
            if key == ord('c'):
                print("[Vision] Starting calibration...")
                tracker.is_calibrated = False
                tracker.calibration_samples = []
                self.is_calibrating = True
            elif key == 27: # ESC
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def broadcast_calibration(self):
        data = network_utils.pack_control(1) # 1 = Start Calibration
        for ip, sock in self.client_tcp_sockets.items():
            try:
                sock.sendall(data)
                print(f"[Control] Sent calibration command to {ip}")
            except Exception as e:
                print(f"[Control] Failed to send to {ip}: {e}")

    def on_move(self, x, y):
        if self.last_pos is None:
            self.last_pos = (x, y)
            return

        dx = x - self.last_pos[0]
        dy = y - self.last_pos[1]
        self.last_pos = (x, y)

        if self.active_client_ip and (dx != 0 or dy != 0):
            # Send via UDP
            data = network_utils.pack_move(dx, dy)
            self.udp_sock.sendto(data, (self.active_client_ip, network_utils.UDP_PORT))
            
            # Lock mouse to center of screen to prevent it from wandering off on the host
            # For prototype, we'll just snap it back if we move too far, or just continuously
            # Wait, moving the mouse programmatically will trigger on_move again! 
            # To avoid infinite loops, we would need to ignore programmatic moves.
            # For this simple prototype, we won't lock the mouse. We'll just let it move.

    def on_click(self, x, y, button, pressed):
        if self.active_client_ip:
            if button == mouse.Button.left:
                b_id = 1
            elif button == mouse.Button.right:
                b_id = 2
            elif button == mouse.Button.middle:
                b_id = 3
            else:
                return
            
            data = network_utils.pack_click(b_id, pressed)
            sock = self.client_tcp_sockets.get(self.active_client_ip)
            if sock:
                try:
                    sock.sendall(data)
                except Exception as e:
                    print(f"Error sending click: {e}")

    def on_scroll(self, x, y, dx, dy):
        if self.active_client_ip:
            data = network_utils.pack_scroll(dx, dy)
            sock = self.client_tcp_sockets.get(self.active_client_ip)
            if sock:
                try:
                    sock.sendall(data)
                except Exception as e:
                    print(f"Error sending scroll: {e}")

if __name__ == "__main__":
    import sys
    
    # Default to 0 (built-in webcam) for host
    camera_source = 0
    if len(sys.argv) > 1:
        camera_source = sys.argv[1]
    
    host = HostController(camera_source)
    host.start()
