#!/bin/sh
for N in 2 4 8 16 32; do
    python hodge2.py N=$N
done
