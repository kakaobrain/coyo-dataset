#/bin/bash

# Install img2dataset
pip install --upgrade gcsfs
pip install tensorflow tensorflow_io
pip install img2dataset



# Setting a Knot Resolver
# https://github.com/rom1504/img2dataset#setting-up-a-knot-resolver
#wget https://secure.nic.cz/files/knot-resolver/knot-resolver-release.deb
#sudo dpkg -i knot-resolver-release.deb
#sudo apt update
#sudo apt install -y knot-resolver
#sudo sh -c 'echo `hostname -I` `hostname` >> /etc/hosts'
#sudo sh -c 'cat /etc/resolv.conf | grep -v nameserver | tee /etc/resolv.conf'
#sudo sh -c 'echo nameserver 127.0.0.1 | tee -a /etc/resolv.conf'
#sudo systemctl stop systemd-resolved
#sudo systemctl start kresd@{1..4}.service
