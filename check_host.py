import socket
import network_utils

def check():
    hostname = socket.gethostname()
    try:
        ips = socket.gethostbyname_ex(hostname)[2]
    except:
        ips = [socket.gethostbyname(hostname)]
        
    print(f"--- Host Diagnostic ---")
    print(f"Hostname: {hostname}")
    print(f"Detected IPs: {ips}")
    
    found_target = False
    for ip in ips:
        if ip.startswith("192.168.1."):
            print(f"Found potential local IP: {ip}")
            found_target = True
            
    if not found_target:
        print("WARNING: No 192.168.1.x IP detected. Ensure you are on the same WiFi as the client.")
    
    print("\nChecking if ports are available to bind:")
    for name, port in [("UDP", network_utils.UDP_PORT), ("TCP", network_utils.TCP_PORT), ("GAZE", network_utils.GAZE_PORT)]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM if name != "UDP" else socket.SOCK_DGRAM)
        try:
            s.bind(('0.0.0.0', port))
            print(f"  [OK] {name} Port {port} is AVAILABLE.")
        except Exception as e:
            print(f"  [X] {name} Port {port} is BUSY or BLOCKED: {e}")
        finally:
            s.close()

if __name__ == "__main__":
    check()
