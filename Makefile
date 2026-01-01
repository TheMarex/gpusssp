all: gpusssp

delta_step.spv: src/delta_step.comp
	glslangValidator -V $< -o $@

gpusssp: src/gpusssp.cpp delta_step.spv
	g++ -std=c++20 $< -lvulkan -o $@

