ifeq ($(platform),)
	platform = unix
	ifeq ($(shell uname -a),)
		platform = win
	else ifneq ($(findstring MINGW,$(shell uname -a)),)
		platform = win
	else ifneq ($(findstring Darwin,$(shell uname -a)),)
		platform = osx
		arch = intel
	ifeq ($(shell uname -p),powerpc)
		arch = ppc
	endif
	else ifneq ($(findstring win,$(shell uname -a)),)
		platform = win
	endif
endif

# system platform
system_platform = unix
ifeq ($(shell uname -a),)
	EXE_EXT = .exe
	system_platform = win
else ifneq ($(findstring MINGW,$(shell uname -a)),)
	system_platform = win
endif

TARGET_NAME = midiviz

ifeq ($(platform), unix)
   TARGET := $(TARGET_NAME)_libretro.so
   fpic := -fPIC
   SHARED := -shared -Wl,--version-script=link.T -Wl,--no-undefined
   LIBS += -lsndfile
else
   CC = gcc
   TARGET := $(TARGET_NAME)_libretro.dll
   SHARED := -shared -static-libgcc -static-libstdc++ -s -Wl,--version-script=link.T -Wl,--no-undefined
endif

ifeq ($(DEBUG), 1)
   CFLAGS += -O0 -g
   CXXFLAGS += -O0 -g
else
   CFLAGS += -O3
   CXXFLAGS += -O3
endif

CFLAGS += -std=gnu99 -I.
CXXFLAGS += -std=gnu++11 -I.
OBJECTS := libretro.o vulkan/vulkan_symbol_wrapper.o midi.o
DEPS := $(OBJECTS:.o=.d)
CFLAGS += -Wall -pedantic $(fpic)
CXXFLAGS += -Wall -pedantic $(fpic)

all: $(TARGET)

-include $(DEPS)

$(TARGET): $(OBJECTS)
	$(CXX) $(fpic) $(SHARED) $(INCLUDES) -o $@ $(OBJECTS) $(LIBS) -lm $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $< -MMD

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< -MMD

clean:
	rm -f $(OBJECTS) $(TARGET) $(DEPS)

.PHONY: clean

