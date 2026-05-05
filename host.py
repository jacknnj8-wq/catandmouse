import socket
import threading
import time
from pynput import mouse
import network_utils

class HostController:
    def __init__(self):
        self.active_client_ip = None
        
        # Sockets
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # TCP Server for Clicks
        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp_server.bind(('0.0.0.0', network_utils.TCP_PORT))
        self.tcp_server.listen(5)
        
        # TCP Server for Gaze State
        self.gaze_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.gaze_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.gaze_server.bind(('0.0.0.0', network_utils.GAZE_PORT))
        self.gaze_server.listen(5)

        self.client_tcp_sockets = {} # IP -> socket

        # Mouse tracking for deltas
        self.last_pos = None
        self.mouse_controller = mouse.Controller()

    def start(self):
        print("Starting Host Server...")
        threading.Thread(target=self.accept_tcp_clients, daemon=True).start()
        threading.Thread(target=self.accept_gaze_clients, daemon=True).start()
        
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
                
                if is_looking:
                    if self.active_client_ip != ip:
                        print(f"[*] Focus switched to client {ip}")
                        self.active_client_ip = ip
                else:
                    if self.active_client_ip == ip:
                        print(f"[*] Focus released by client {ip}")
                        self.active_client_ip = None
        except Exception as e:
            print(f"[Gaze] Error with {ip}: {e}")
        finally:
            client.close()
            if self.active_client_ip == ip:
                self.active_client_ip = None

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
    host = HostController()
    host.start()
