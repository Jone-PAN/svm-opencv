#!/usr/bin/env bash
g++ -std=c++11 cpp_test.cpp -o cpp_test `pkg-config --cflags --libs opencv` -pthread 


