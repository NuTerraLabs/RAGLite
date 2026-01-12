# HTTP vs HTTPS — What’s the difference?

**HTTP (Hypertext Transfer Protocol)** is an application-layer protocol for the web. It defines methods like **GET**, **POST**, **PUT**, **DELETE** and uses status codes (e.g., **200 OK**, **404 Not Found**, **500 Internal Server Error**).

**HTTPS** wraps HTTP inside **TLS** to provide encryption, server authentication, and integrity. The **TLS handshake** negotiates keys and ciphers, enabling confidentiality and protecting against on-path attackers.

Other useful ideas:
- **Idempotent** methods: GET, PUT, DELETE (repeated calls should have the same effect).
- **Safe** methods: GET, HEAD (shouldn’t change server state).
- Persistent connections, HTTP/2 multiplexing, HTTP/3 over QUIC.
