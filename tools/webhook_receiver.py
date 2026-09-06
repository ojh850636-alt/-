from http.server import BaseHTTPRequestHandler, HTTPServer


class Receiver(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length)
        print("Received webhook:", body.decode("utf-8"))
        self.send_response(200)
        self.end_headers()


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 9001), Receiver)
    print("Webhook receiver running on http://0.0.0.0:9001")
    server.serve_forever()
