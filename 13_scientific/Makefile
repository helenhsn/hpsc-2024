final_report:
	$(info "0 - USING GLOBAL MEMORY ONLY [Evaluation - SEQUENTIAL]")
	g++ 41x41_cavity.cpp -o cuda_cavity

	$(info "1 - USING GLOBAL MEMORY ONLY [Evaluation - PARALLEL]")
	nvcc 41x41_cavity.cu -o  cuda_cavity_GM
	

	$(info "2 - USING SHARED MEMORY & GLOBAL MEMORY [Additional work]")
	nvcc 32x32_cavity.cu -o  cuda_cavity_SM
