from http.server import BaseHTTPRequestHandler
import json

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        response = {
            "message": "Smart Attendance API is live!",
        }
        self.wfile.write(json.dumps(response).encode())

# This is the handler that Vercel will use to serve your app
def handler(event, context):
    return Handler(event, context)
