from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Smart Attendance System</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    background-color: #f0f2f5;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }
                .card {
                    background: white;
                    padding: 2.5rem;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    text-align: center;
                    max-width: 400px;
                    width: 90%;
                }
                h1 { color: #1a1a1a; margin-bottom: 0.5rem; font-size: 1.8rem; }
                p { color: #666; margin-bottom: 2rem; line-height: 1.5; }
                .btn {
                    display: inline-block;
                    background-color: #0070f3;
                    color: white;
                    padding: 12px 24px;
                    border-radius: 6px;
                    text-decoration: none;
                    font-weight: 600;
                    transition: background-color 0.2s;
                }
                .btn:hover { background-color: #0051a2; }
                .status { 
                    display: inline-block;
                    margin-top: 1.5rem;
                    font-size: 0.875rem;
                    color: #2e7d32;
                    background: #e8f5e9;
                    padding: 4px 12px;
                    border-radius: 100px;
                }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Smart Attendance 📸</h1>
                <p>
                    Advanced Face Recognition Attendance System.<br>
                    This is the Cloud API endpoint.
                </p>
                
                <a href="https://github.com/Karthigaiselvam-R-official/Smart_Attendance" class="btn">
                    Download on GitHub
                </a>
                
                <br>
                <div class="status">● System Operational</div>
            </div>
        </body>
        </html>
        """
        self.wfile.write(html.encode('utf-8'))
