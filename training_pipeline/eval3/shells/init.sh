#!/bin/bash
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 2222/tcp
ufw allow in from 0.0.0.0/0 to any port 22
ufw --force enable
iptables -F DOCKER-USER
iptables -A DOCKER-USER -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A DOCKER-USER -i docker0 ! -o docker0 -j ACCEPT
iptables -A DOCKER-USER -i br+     ! -o br+     -j ACCEPT
iptables -A DOCKER-USER -i cni+    ! -o cni+    -j ACCEPT
iptables -A DOCKER-USER -i cali+   ! -o cali+   -j ACCEPT
iptables -A DOCKER-USER -i docker0 -o docker0 -j ACCEPT
iptables -A DOCKER-USER -i br+     -o br+     -j ACCEPT
iptables -A DOCKER-USER -i cni+    -o cni+    -j ACCEPT
iptables -A DOCKER-USER -i cali+   -o cali+   -j ACCEPT
iptables -A DOCKER-USER -i lo -j ACCEPT
iptables -A DOCKER-USER -p tcp --dport 22 -j ACCEPT
iptables -A DOCKER-USER -j DROP
iptables -A DOCKER-USER -j RETURN
