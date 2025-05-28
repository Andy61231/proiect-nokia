from waitress import serve
from main import app  

if __name__ == '__main__':
    host = '0.0.0.0'
    port = 80
    print(f"Serverul Waitress pornește pe http://{host}:{port}")
    print(f"Dacă ai configurat port forwarding, încearcă: http://proiect-nokia.duckdns.org:{port}")
    print("Sau, dacă portul extern este 80: http://proiect-nokia.duckdns.org")
    serve(app, host=host, port=port)
