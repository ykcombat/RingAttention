run_alone: build_standalone
	nvshmrun -n 2 ./build/ring_attention

build_standalone:
	cmake -B build
	cmake --build build

