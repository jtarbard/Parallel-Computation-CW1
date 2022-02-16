#
# Very simple makefile for coursework 1. You are free to update this if you
# wish to add files etc., but ensure (a) the executable name is unchanged,
# (b) it still works on a school Linux machine, and (c) the file cwk1_extra.h
# is included and unaltered. Also make sure that any new files are in the same
# directory (i.e. do not use subdirectories).
#
EXE = cwk1
CC = gcc
CCFLAGS = -Wall -fopenmp

all:
	$(CC) $(CCFLAGS) -o $(EXE) cwk1.c

