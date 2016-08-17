VERT_SHADERS := $(wildcard *.vert)
FRAG_SHADERS := $(wildcard *.frag)
COMP_SHADERS := $(wildcard *.comp)
SPIRV := \
	$(VERT_SHADERS:.vert=.vert.inc) \
	$(FRAG_SHADERS:.frag=.frag.inc) \
	$(COMP_SHADERS:.comp=.comp.inc)
DEPS := $(SPIRV:.inc=.inc.d)

GLSLANG := glslc
GLSLFLAGS := -mfmt=c -MD

all: $(SPIRV)

-include $(DEPS)

%.frag.inc: %.frag
	$(GLSLANG) $(GLSLFLAGS) -o $@ $<

%.comp.inc: %.comp
	$(GLSLANG) $(GLSLFLAGS) -o $@ $<

%.vert.inc: %.vert
	$(GLSLANG) $(GLSLFLAGS) -o $@ $<

clean:
	rm -f $(SPIRV) $(DEPS)

.PHONY: clean
