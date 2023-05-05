import http.server
import socketserver
import os
# command:
# python -m http.server --directory web

PORT = 8000
DIRECTORY = "web"
web_dir = os.path.join(os.path.dirname(__file__), 'web')
os.chdir(web_dir)

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)


with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()