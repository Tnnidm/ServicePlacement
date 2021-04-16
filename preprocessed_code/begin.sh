#!/usr/bin/env bash

python preprocessing_car.py \
 --filedir '../../data/' \
 --dataset 'chengdushi_1001_1010.csv' &

python preprocessing_car.py \
 --filedir '../../data/' \
 --dataset 'chengdushi_1010_1020.csv' &

python preprocessing_car.py \
 --filedir '../../data/' \
 --dataset 'chengdushi_1020_1031.csv' &

wait
echo "preprocessing_car.py done!"

python preprocessing_car_2.py

echo "preprocessing_car_2.py done!"