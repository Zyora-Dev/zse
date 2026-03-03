#!/bin/bash
# ZSE Website Deployment Script for DigitalOcean
# Domain: zllm.in
# Server: 45.55.51.77

set -e

SSH_KEY="/Users/redfoxhotels/amd"
SERVER="root@45.55.51.77"
DOMAIN="zllm.in"

echo "=== ZSE Website Deployment ==="
echo "Server: $SERVER"
echo "Domain: $DOMAIN"
echo ""

# Step 1: Build the website locally
echo "[1/6] Building website locally..."
cd /Users/redfoxhotels/zse/website
npm run build

# Step 2: Install dependencies on server
echo "[2/6] Setting up server..."
ssh -i $SSH_KEY $SERVER << 'ENDSSH'
# Update system
apt-get update -y

# Install Node.js 20.x if not present
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi

# Install PM2 globally
npm install -g pm2

# Install Nginx if not present
if ! command -v nginx &> /dev/null; then
    apt-get install -y nginx
fi

# Install certbot for SSL
if ! command -v certbot &> /dev/null; then
    apt-get install -y certbot python3-certbot-nginx
fi

# Create app directory
mkdir -p /var/www/zllm
ENDSSH

# Step 3: Upload website files
echo "[3/6] Uploading website files..."
rsync -avz --delete \
    -e "ssh -i $SSH_KEY" \
    --exclude 'node_modules' \
    --exclude '.git' \
    /Users/redfoxhotels/zse/website/ \
    $SERVER:/var/www/zllm/

# Step 4: Install dependencies and start app on server
echo "[4/6] Installing dependencies and starting app..."
ssh -i $SSH_KEY $SERVER << 'ENDSSH'
cd /var/www/zllm
npm install --production
pm2 delete zllm 2>/dev/null || true
pm2 start npm --name "zllm" -- start
pm2 save
pm2 startup systemd -u root --hp /root 2>/dev/null || true
ENDSSH

# Step 5: Configure Nginx
echo "[5/6] Configuring Nginx..."
ssh -i $SSH_KEY $SERVER << ENDSSH
cat > /etc/nginx/sites-available/zllm << 'EOF'
server {
    listen 80;
    server_name zllm.in www.zllm.in;

    location / {
        proxy_pass http://127.0.0.1:5278;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

ln -sf /etc/nginx/sites-available/zllm /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true
nginx -t && systemctl reload nginx
ENDSSH

# Step 6: Setup SSL (optional - requires DNS to be pointed)
echo "[6/6] Setting up SSL..."
echo "Make sure DNS for $DOMAIN points to 45.55.51.77 before running SSL setup"
read -p "Is DNS configured? (y/n): " dns_ready
if [ "$dns_ready" = "y" ]; then
    ssh -i $SSH_KEY $SERVER "certbot --nginx -d zllm.in -d www.zllm.in --non-interactive --agree-tos -m admin@zllm.in"
else
    echo "Skipping SSL setup. Run this later:"
    echo "ssh -i $SSH_KEY $SERVER \"certbot --nginx -d zllm.in -d www.zllm.in\""
fi

echo ""
echo "=== Deployment Complete ==="
echo "Website: http://zllm.in (or https:// if SSL was configured)"
echo ""
