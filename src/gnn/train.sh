# nohup python -u train.py -m gin0jk -f 504 -l 5 -hs 128 -b 20 -g 0 > log_train 2>&1 &
nohup python -u train.py -m dgcnn -f 504 -b 20 -g 0 > log_train 2>&1 &