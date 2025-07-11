#!/bin/bash

#     "twitter.com"
#    "www.twitter.com"
#    "x.com"
#    "www.x.com"
BLOCKLIST=(

)

HOSTS_FILE="/etc/hosts"
BACKUP_FILE="/etc/hosts.backup"

# Backup the original hosts file if not already backed up
if [ ! -f "$BACKUP_FILE" ]; then
    sudo cp $HOSTS_FILE $BACKUP_FILE
fi

# Remove existing blocked entries
sudo sed -i '' '/# Blocked by script/d' $HOSTS_FILE
for DOMAIN in "${BLOCKLIST[@]}"; do
    sudo sed -i '' "/$DOMAIN/d" $HOSTS_FILE
done

# Add new blocked entries
for DOMAIN in "${BLOCKLIST[@]}"; do
    echo "127.0.0.1 $DOMAIN # Blocked by script" | sudo tee -a $HOSTS_FILE > /dev/null
done

# Flush DNS cache to apply changes
sudo dscacheutil -flushcache
sudo killall -HUP mDNSResponder

echo "Websites blocked: ${BLOCKLIST[*]}"
