all: gpusssp

delta_step.spv: delta_step.comp
	glslangValidator -V delta_step.comp -o delta_step.spv

gpusssp: main.cpp delta_step.spv
	g++ -std=c++20 main.cpp -lvulkan -o $@

