#!/bin/bash

for i in *
do
  mv -v "$i" "c${i}"
done
