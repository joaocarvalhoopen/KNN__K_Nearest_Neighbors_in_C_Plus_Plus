# Notes:
#      To compile for release with MINGW32 do "mingw32-make".
#      To compile for debug with MINGW32 do "mingw32-make TARGET=debug".

# Release version.
CFLAGS = -O2 -Wall

# To compile for debug call with "mingw32-make TARGET=debug" 
ifeq "$(strip $(filter debug,$(TARGET)))" "debug"
# debug
CFLAGS = -g -Wall
endif

## To compile for release call with "mingw32.make TARGET=release" 
#ifeq "$(strip $(filter release,$(TARGET)))" "release"
## Release version.
#CFLAGS = -O2 -Wall
#endif

CC = g++

SRCS = KNN__K_Nearest_Neighbors.cpp
PROG = KNN__K_Nearest_Neighbors.exe

$(PROG) : $(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) 


clean: 
	del *.o KNN__K_Nearest_Neighbors.exe

