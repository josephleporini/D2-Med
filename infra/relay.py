import http.server, subprocess, os, json, urllib.parse
R='/workspace'
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(s):
        u=urllib.parse.urlparse(s.path); q=urllib.parse.parse_qs(u.query)
        if u.path=='/ready':
            ok=os.path.exists('/tmp/ready'); s.send_response(200 if ok else 503); s.end_headers(); s.wfile.write(b'ok' if ok else b'wait'); return
        if u.path=='/list':
            b=json.dumps(sorted(os.listdir(R))).encode(); s.send_response(200); s.end_headers(); s.wfile.write(b); return
        if u.path=='/stat':
            d=q['d'][0]; n=subprocess.run(f"find '{R}/{d}' -type f | wc -l; du -sb '{R}/{d}' | cut -f1",shell=True,capture_output=True,text=True).stdout.split()
            s.send_response(200); s.end_headers(); s.wfile.write(json.dumps(n).encode()); return
        if u.path=='/tar':
            d=q['d'][0]; s.send_response(200); s.send_header('Content-Type','application/x-tar'); s.end_headers()
            p=subprocess.Popen(['tar','cf','-','-C',R,d],stdout=subprocess.PIPE)
            for c in iter(lambda:p.stdout.read(1<<20),b''): s.wfile.write(c)
            p.wait(); return
        s.send_response(404); s.end_headers()
    def log_message(s,*a): pass
http.server.ThreadingHTTPServer(('0.0.0.0',8000),H).serve_forever()
