/************************************************************************
 *
 * NMF-mGPU - Non-negative Matrix Factorization on multi-GPU systems.
 *
 * Copyright (C) 2011-2015:
 *
 *	Edgardo Mejia-Roa(*), Carlos Garcia(*), Jose Ignacio Gomez(*),
 *	Manuel Prieto(*), Francisco Tirado(*) and Alberto Pascual-Montano(**).
 *
 *	(*)  ArTeCS Group, Complutense University of Madrid (UCM), Spain.
 *	(**) Functional Bioinformatics Group, Biocomputing Unit,
 *		National Center for Biotechnology-CSIC, Madrid, Spain.
 *
 *	E-mail for E. Mejia-Roa: <edgardomejia@fis.ucm.es>
 *	E-mail for A. Pascual-Montano: <pascual@cnb.csic.es>
 *
 *
 * This file is part of NMF-mGPU.
 *
 * NMF-mGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * NMF-mGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with NMF-mGPU. If not, see <http://www.gnu.org/licenses/>.
 *
 ***********************************************************************/
/**********************************************************
 *
 * NMF_GPU.c
 *	Main program for single-GPU systems.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Additional information:
 *		NMFGPU_VERBOSE: Shows some messages concerning the progress of the program, as well as
 *				some configuration parameters.
 *		NMFGPU_VERBOSE_2: Shows the parameters in some routine calls.
 *
 *	CPU timing:
 *		NMFGPU_PROFILING_GLOBAL: Compute total elapsed time. If GPU time is NOT being computed,
 *					the CPU thread performs active waiting (i.e., spins) on
 *					synchronization calls, such as cudaDeviceSynchronize() or
 *					cudaStreamSynchronize(). Otherwise, the CPU thread is blocked.
 *
 *	GPU timing (WARNING: They PREVENT asynchronous operations. The CPU thread is blocked on synchronization):
 *		NMFGPU_PROFILING_TRANSF: Compute timing of data transfers. Shows additional information.
 *		NMFGPU_PROFILING_KERNELS: Compute timing of CUDA kernels. Shows additional information.
 *
 *	Debug / Testing:
 *		NMFGPU_CPU_RANDOM: Uses the CPU (host) random generator (not the CURAND library).
 *		NMFGPU_FIXED_INIT: Initializes W and H with "random" values generated from a fixed seed (defined in common.h).
 *		NMFGPU_DEBUG: Shows the result of each matrix operation and data transfer.
 *		NMFGPU_FORCE_BLOCKS: Forces the processing of the input matrix as four blocks.
 *				     It also disables mapping of host memory into device address space.
 *		NMFGPU_TEST_BLOCKS: Just shows block information structure. No GPU memory is allocated.
 *
 **********************************************************
 **********************************************************
 **********************************************************
 *
 * Data matrices:
 *	V (N rows, M columns): input matrix
 *	W (N,K): output matrix
 *	H (K,M): output matrix,
 * such that: V  ~  W * H.
 *
 * Arguments:
 *	Matrix V (and its dimensions)
 *	K: Factorization Rank
 *
 *
 * NOTE: In order to improve performance:
 *
 *	+ Matrix H is stored in memory as COLUMN-major (i.e., it is transposed).
 *
 *	+ All matrices include useless data for padding. Padded dimensions
 *	  are denoted with the 'p' character. For instance:
 *		Mp, which is equal to <M + padding>
 *		Kp, which is equal to <K + padding>.
 *
 *	  Data alignment is controlled by the global variable: memory_alignment.
 *
 *	  This leads to the following limits:
 *		- Maximum number of columns (padded or not): matrix_max_pitch.
 *		- Maximum number of rows: matrix_max_non_padded_dim.
 *		- Maximum number of items: matrix_max_num_items.
 *
 *	  All four GLOBAL variables must be initialized with the set_matrix_limits()
 *	  function.
 *
 ****************
 *
 * Matrix tags:
 *
 * Any matrix may include the following "tag" elements:
 *
 *	+ A short description string, referred as "name".
 *	+ A list of column headers.
 *	+ A list of row labels.
 *
 * Each list is stored in a "struct tag_t" structure, which is composed by:
 *	+ All tokens stored as a (large) single string.
 *	+ An array of pointers to such tokens.
 *
 * All three elements (the "name" string, and the two tag_t structures) are
 * then stored in a "struct matrix_tags_t" structure.
 *
 * Both types of structure are defined in "matrix_io_routines.h".
 *
 ****************
 *
 * Multi-GPU version:
 *
 * When the input matrix V is distributed among multiple devices each host thread processes
 * the following sets of rows and columns:
 *
 *	Vrow[ 1..NpP ][ 1..M ] <-- V[ bN..(bN+NpP) ][ 1..M ]	(i.e., NpP rows, starting from bN)
 *	Vcol[ 1..N ][ 1..MpP ] <-- V[ 1..N ][ bM..(bM+MpP) ]	(i.e., MpP columns, starting from bM)
 *
 * Such sets allow to update the corresponding rows and columns of W and H, respectively.
 *
 * Note that each host thread has a private full copy of matrices W and H, which must be synchronized
 * after being updated.
 *
 ****************
 *
 * Large input matrix (blockwise processing):
 *
 * If the input matrix (or the portion assigned to this device) is too large for the GPU memory,
 * it must be blockwise processed as follow:
 *
 *	d_Vrow[1..BLN][1..Mp] <-- Vrow[ offset..(offset + BLN) ][1..Mp]			(i.e., BLN <= NpP rows)
 *	d_Vcol[1..N][1..BLMp] <-- Vcol[1..N][ offset_Vcol..(offset_Vcol + BLMp) ]	(i.e., BLM <= MpP columns)
 *
 * Note that padded dimensions are denoted with the suffix 'p' (e.g., Mp, BLMp, etc).
 *
 * In any case, matrices W and H are fully loaded into the GPU memory.
 *
 * Information for blockwise processing is stored in two block_t structures (one for each dimension).
 * Such structures ('block_N' and 'block_M') are initialized in the init_block_conf() routine.
 *
 ****************
 *
 * Mapped Memory on integrated GPUs:
 *
 * On integrated systems, such as notebooks, where device memory and host memory are physically the
 * same (but disjoint regions), any data transfer between host and device memory is superfluous.
 * In such case, host memory is mapped into the address space of the device, and all transfer
 * operations are skipped. Memory for temporary buffers (e.g., d_WH or d_Aux) is also allocated
 * on the HOST and then mapped. This saves device memory, which is typically required for graphics/video
 * operations.
 *
 * This feature is disabled if NMFGPU_FORCE_BLOCKS is non-zero.
 *
 *********************************************************/
#ifndef NMF_GPU_H_
#define NMF_GPU_H_

#include "NMF_routines.h"
#include "matrix_operations.h"
#include "GPU_setup.h"
#include "matrix_io.h"
#include "matrix_io_routines.h"
#include "common.h"
#include "index_type.h"
#include "real_type.h"

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>	/* uintptr_t */
#include <stdbool.h>
#include <stdlib.h>

static int nmf( index_t nIters, index_t niter_test_conv, index_t stop_threshold );
void bionmfSingleGPU(float* W, float* H, float* V, unsigned long M, unsigned int N, unsigned int K, unsigned int J, unsigned int T, unsigned int I);


#endif
