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

#include "NMF_GPU.h"

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Global variables */

#if NMFGPU_DEBUG || NMFGPU_VERBOSE || NMFGPU_VERBOSE_2
	static bool const dbg_shown_by_all = false;	// Information or error messages on debug.
	static bool const verb_shown_by_all = false;	// Information messages in verbose mode.
#endif
static bool const shown_by_all = false;			// Information messages.
static bool const sys_error_shown_by_all = false;	// System error messages.
static bool const error_shown_by_all = false;		// Error messages on invalid arguments or I/O data.

static int nmf( index_t nIters, index_t niter_test_conv, index_t stop_threshold ){

	#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
		int status = EXIT_SUCCESS;
	#endif

	pBLN = 0;	// Current index in block_N.xxx[].
	pBLM = 0;	// Current index in block_M.xxx[].

	stepN = 1;	// Loop directions: +1 (forward) || -1 (backward).
	stepM = 1;	// Loop directions: +1 (forward) || -1 (backward).

	psNMF_N = 0;	// Current index in streams_NMF[].
	psNMF_M = 0;	// Current index in streams_NMF[].

	colIdx = 0;	// Current column index in Vcol. It corresponds to <bM + colIdx> in H and d_H.
	rowIdx = 0;	// Current row index in Vrow. It corresponds to <bN + rowIdx> in W and d_W.


	#if NMFGPU_PROFILING_GLOBAL
		// GPU time
		struct timeval gpu_tv;
		gettimeofday( &gpu_tv, NULL );
	#endif

	// ----------------------------

	// Initializes the random-number generator.
	{
		index_t const seed = get_seed();

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			init_random( seed );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				exit(-1);
		#endif
	}

	// ----------------------------

	// Initializes matrix W

	#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
		status =
	#endif

		set_random_values( d_W, N, K, Kp,
					#if NMFGPU_CPU_RANDOM
						W,
					#endif
					#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2
						false,		// NO matrix transposing
					#endif
					#if NMFGPU_CPU_RANDOM && (NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
						|| (! NMFGPU_PROFILING_GLOBAL))
						"W",
					#endif
					#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 || (NMFGPU_CPU_RANDOM && NMFGPU_DEBUG_TRANSF) \
						|| ((! NMFGPU_CPU_RANDOM) && (! NMFGPU_PROFILING_GLOBAL))
						"d_W",
					#endif
					#if ( NMFGPU_CPU_RANDOM && NMFGPU_PROFILING_TRANSF )
						&upload_W_timing,
					#endif
					stream_W, &event_W );

	#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
		if ( status != EXIT_SUCCESS ) {
			destroy_random();
			exit(-1);
		}
	#endif

	// ----------------------------

	// Initializes matrix H
	{

		// Offset to the portion of data initialized by this process.
		size_t const offset = (size_t) bM * (size_t) Kp;
		real *const p_dH = &d_H[ offset ];
		#if NMFGPU_CPU_RANDOM
			real *const pH = &H[ offset ];
		#endif

		// Number of rows
		index_t const height = MpP;

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			set_random_values( p_dH, height, K, Kp,
					#if NMFGPU_CPU_RANDOM
						pH,
					#endif
					#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 || (NMFGPU_CPU_RANDOM && NMFGPU_DEBUG_TRANSF)
						true,		// Matrix transposing
					#endif
					#if NMFGPU_CPU_RANDOM && (NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
						|| (! NMFGPU_PROFILING_GLOBAL))
						"H",
					#endif
					#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 || (NMFGPU_CPU_RANDOM && NMFGPU_DEBUG_TRANSF) \
						|| ((! NMFGPU_CPU_RANDOM) && (! NMFGPU_PROFILING_GLOBAL))
						"d_H",
					#endif
					#if ( NMFGPU_CPU_RANDOM && NMFGPU_PROFILING_TRANSF )
						&upload_H_timing,
					#endif
					streams_NMF[ psNMF_M ], NULL );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS ) {
				destroy_random();
				exit(-1);
			}
		#endif
	}

	// ----------------------------

	// Finalizes the random-number generator.

	destroy_random();

	// ----------------------------

	// Uploads matrix V
	{
		// Block configuration.
		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
			index_t const BLM = block_M.BL[ pBLM ];		// Number of columns.
		#endif
		index_t const BLMp = block_M.BLp[ pBLM ];		// Number of columns (with padding).
		index_t const BLN  = block_N.BL[ pBLN ];		// Number of rows.

		// d_Vcol
		if ( d_Vcol != d_Vrow ) {

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				status =
			#endif

				upload_matrix_partial( Vcol, N, MpPp, 0, colIdx,	// Starting row: 0
							#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
								BLM,
							#endif
							#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
								|| (! NMFGPU_PROFILING_GLOBAL)
								"Vcol",
							#endif
							#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
								"d_Vcol",
							#endif
							BLMp, d_Vcol, stream_Vcol, event_Vcol
							#if NMFGPU_PROFILING_TRANSF
								, &upload_Vcol_timing
							#endif
							);

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					exit(-1);
			#endif

		} // if d_Vcol != d_Vrow

		// ----------------------------

		// d_Vrow

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			upload_matrix_partial( Vrow, BLN, Mp, rowIdx, 0,	// Starting column: 0
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							M,
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
							|| (! NMFGPU_PROFILING_GLOBAL)
							"Vrow",
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							"d_Vrow",
						#endif
						Mp, d_Vrow, stream_Vrow, event_Vrow
						#if NMFGPU_PROFILING_TRANSF
							, &upload_Vrow_timing
						#endif
						);

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				exit(-1);
		#endif

	} // uploads input matrix.

	// ----------------------------

	// Number of iterations

	index_t const niter_conv = (nIters / niter_test_conv);		// Number of times to perform test of convergence.
	index_t const niter_rem  = (nIters % niter_test_conv);		// Remaining iterations.
	bool converged = false;

	#if NMFGPU_VERBOSE
		print_message( verb_shown_by_all, "niter_test_conv=%" PRI_IDX ", niter_conv=%" PRI_IDX ", niter_rem=%" PRI_IDX ".\n",
				niter_test_conv, niter_conv, niter_rem );
	#endif

	print_message( shown_by_all, "Starting NMF( K=%"PRI_IDX" )...\n", K );
	flush_output( false );

	// ------------------------

	index_t inc = 0;	// Number of it. w/o changes.

	/* Performs all <nIters> iterations in <niter_conv> groups
	 * of <niter_test_conv> iterations each.
	 */

	index_t iter = 0;	// Required outside this loop.

	for ( ; iter<niter_conv ; iter++ ) {

		// Runs NMF for niter_test_conv iterations...
		for ( index_t i=0 ; i<niter_test_conv ; i++ ) {

			#if NMFGPU_DEBUG
			///////////////////////////////
				print_message( verb_shown_by_all, "\n\n============ iter=%" PRI_IDX ", Loop %" PRI_IDX
						" (niter_test_conv): ============\n------------ Matrix H: ------------\n", iter,i);
			/////////////////////////////
			#endif

			/*
			 * WH(N,BLMp) = W * pH(BLM,Kp)
			 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
			 * Haux(BLM,Kp) = W' * WH(N,BLMp)
			 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W
			 */

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				status =
			#endif

				update_H();

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					exit(-1);
			#endif


			#ifdef NMFGPU_DEBUG
			///////////////////////////////
				print_message( verb_shown_by_all, "\n------------ iter=%i, loop %i (niter_test_conv) Matrix W: ------------\n",
						iter,i);
			/////////////////////////////
			#endif

			/*
			 * WH(BLN,Mp) = W(BLN,Kp) * H
			 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
			 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
			 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
			 */

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				status =
			#endif

				update_W();

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					exit(-1);
			#endif


		} // for niter_test_conv times.

		// -------------------------------------

		// Adjusts matrices W and H.

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			matrix_adjust( d_H, M, Kp,
					#if NMFGPU_DEBUG
						K, true, 	// Matrix transposing
					#endif
					#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
						"d_H",
					#endif
					stream_H, NULL );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				exit(-1);
		#endif

		// ----------------------------

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			matrix_adjust( d_W, N, Kp,
					#if NMFGPU_DEBUG
						K, false,	// No matrix transposing
					#endif
					#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
						"d_W",
					#endif
					stream_W, &event_W );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				exit(-1);
		#endif

		// -------------------------------------

		// Test of convergence

		// Computes classification vector

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			get_classification( d_classification, classification );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				exit(-1);
		#endif

		// ----------------------------

		// Computes differences

		size_t const diff = get_difference( classification, last_classification, M );

			#ifdef NMFGPU_DEBUG
			///////////////////////////////
				print_message( dbg_shown_by_all, "\nReturned difference between classification vectors: %zu\n", diff );
			/////////////////////////////
			#endif

		// -------------------------------------

		// Saves the new classification.
		{
			// It just swaps the pointers.
			index_t *restrict const h_tmp = classification;
			classification = last_classification;
			last_classification = h_tmp;

			/* If host memory was mapped into the address space of the device,
			 * pointers in device memory must also be swapped.
			 */
			if ( mappedHostMemory ) {
				index_t *restrict const d_tmp = d_classification;
				d_classification = d_last_classification;
				d_last_classification = d_tmp;
			}
		}

		// Stops if Connectivity matrix (actually, the classification vector) has not changed over last <stop_threshold> iterations.

		if ( diff )
			inc = 0;	// Restarts counter.

		// Increments the counter.
		else if ( inc < stop_threshold )
			inc++;

		#if ! NMFGPU_DEBUG
		// Algorithm has converged.
		else {
			iter++; // Adds to counter the last <niter_test_conv> iterations performed
			converged = true;
			break;
		}
		#endif

	} // for  ( nIters / niter_test_conv ) times

	// ---------------------------------------------------------

	// Remaining iterations (if NMF has not converged yet).

	if ( (!converged) * niter_rem ) { // (converged == false) && (niter_rem > 0)

		#if NMFGPU_VERBOSE
			print_message( verb_shown_by_all, "\nPerforming remaining iterations (%" PRI_IDX ")...\n", niter_rem);
		#endif

		// Runs NMF for niter_rem iterations...
		for ( index_t i=0 ; i<niter_rem ; i++ ) {

			#ifdef NMFGPU_DEBUG
			///////////////////////////////
				print_message( verb_shown_by_all, "\n============ Loop %" PRI_IDX " (remaining) ============\n"
						"------------ Matrix H: ------------\n",i);
			/////////////////////////////
			#endif

			/*
			 * WH(N,BLMp) = W * pH(BLM,Kp)
			 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
			 * Haux(BLM,Kp) = W' * WH(N,BLMp)
			 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W
			 */

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				status =
			#endif

				update_H();

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					exit(-1);
			#endif



			#ifdef NMFGPU_DEBUG
			///////////////////////////////
				print_message(verb_shown_by_all, "\n------------ Matrix W (loop=%" PRI_IDX ",remaining): ------------\n",i);
			/////////////////////////////
			#endif

			/*
			 * WH(BLN,Mp) = W(BLN,Kp) * H
			 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
			 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
			 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
			 */

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				status =
			#endif

				update_W();

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					exit(-1);
			#endif

		} // for niter_rem times.

	} // if has not yet converged.

	#if NMFGPU_VERBOSE
		print_message( verb_shown_by_all, "Done.\n" );
	#endif

	// --------------------------------

	// Number of iterations performed.

	index_t num_iter_performed = nIters;
	if ( converged ) {
		num_iter_performed = iter * niter_test_conv;
		print_message( shown_by_all, "NMF: Algorithm converged in %" PRI_IDX " iterations.\n", num_iter_performed );
	}
	else
		print_message( shown_by_all, "NMF: %" PRI_IDX " iterations performed.\n", num_iter_performed );

	// --------------------------------

	// Downloads output matrices
	{
		bool const real_data = true;
		size_t const data_size = sizeof(real);
		size_t nitems = (size_t) M * (size_t) Kp;

		// d_H

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			download_matrix( H, nitems, data_size, d_H,
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
						M, K, Kp, real_data, true, "H",		// Matrix transposing
					#endif
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
						|| ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_TRANSF))
						"d_H",
					#endif
					#if NMFGPU_PROFILING_TRANSF
						&download_H_timing,
					#endif
					stream_H );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				exit(-1);
		#endif

		// ----------------------------

		// d_W

		nitems = (size_t) N * (size_t) Kp;

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			download_matrix( W, nitems, data_size, d_W,
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
						N, K, Kp, real_data, false, "W",	// NO matrix transposing
					#endif
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
						|| ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_TRANSF))
						"d_W",
					#endif
					#if NMFGPU_PROFILING_TRANSF
						&download_W_timing,
					#endif
					stream_W );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				exit(-1);
		#endif

	}

	// --------------------------------

	/* Checks results:
	 *
	 * Computes the "distance" between V and W*H as follow:
	 *
	 *	distance = norm( V - (W*H) ) / norm( V ),
	 * where
	 *	norm( X ) = sqrt( dot_X )
	 *	dot_V	 <-- dot_product( V, V )
	 *	dot_VWH  <-- dot_product( V-(W*H), V-(W*H) )
	 */

	real dot_V = REAL_C( 0.0 ), dot_VWH = REAL_C( 0.0 );

	#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
		status =
	#endif

		dot_product_VWH( &dot_V, &dot_VWH );

	#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
		if ( status != EXIT_SUCCESS )
			exit(-1);
	#endif

		#if NMFGPU_DEBUG || NMFGPU_VERBOSE
		///////////////////////////////
			print_message( dbg_shown_by_all, "\tnorm(V)=%g, norm(V-WH)=%g\n", SQRTR( dot_V ), SQRTR( dot_VWH ) );
		///////////////////////////////
		#endif

	print_message( shown_by_all, "Distance between V and W*H: %g\n", SQRTR( dot_VWH ) / SQRTR( dot_V ) );

	// --------------------------------


	#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
		status =
	#endif

		check_cuda_status();

	#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
		if ( status != EXIT_SUCCESS )
			exit(-1);
	#endif

	// --------------------------------

	#if NMFGPU_PROFILING_GLOBAL
	{
		// GPU time
		struct timeval gpu_ftv, gpu_etv;
		gettimeofday( &gpu_ftv, NULL );
		timersub( &gpu_ftv, &gpu_tv, &gpu_etv );	// etv = ftv - tv
		float const total_gpu_time = gpu_etv.tv_sec + ( gpu_etv.tv_usec * 1e-06f );
		print_message( shown_by_all, "GPU + classification + check_result time: %g seconds.\n", total_gpu_time );
	}
	#endif

	// ----------------------------

	return EXIT_SUCCESS;

} // nmf
//M = cols N = rows
void bionmfSingleGPU(float* W, float* H, float* V, unsigned long M, unsigned int N, unsigned int K, unsigned int J, unsigned int T, unsigned int I){

	#if NMFGPU_PROFILING_GLOBAL
		// Elapsed time
		struct timeval t_tv;
	#endif

	process_id = 0;		// Global variables.
	num_act_processes = num_processes = 1;

	// Default limits for matrix dimensions. They may be later adjusted, at device initialization.
	set_default_matrix_limits();

	int status = EXIT_SUCCESS;

	// ----------------------------------------

	#if NMFGPU_DEBUG || NMFGPU_DEBUG_READ_MATRIX || NMFGPU_DEBUG_READ_MATRIX2 \
		|| NMFGPU_DEBUG_READ_FILE || NMFGPU_DEBUG_READ_FILE2 || NMFGPU_VERBOSE_2

		// Permanently flushes the output stream in order to prevent losing messages if the program crashes.
		flush_output( true );

	#endif

	// ----------------------------------------

	// Reads all parameters and performs error-checking.

	Kp = get_padding(K);						// Padded factorization rank.
	index_t const nIters = I;			// Maximum number of iterations per run.
	index_t const niter_test_conv = T;	// Number of iterations before testing convergence.
	index_t const stop_threshold = J;	// Stopping criterion.
	index_t const gpu_device = 0;		// Device ID.
	index_t nrows = N, ncols = M, pitch = get_padding(N);
	NpP = N;
	MpP = M;
	MpPp = Mp = pitch;
	bN = 0;
	bM = 0;


	// Compute classification vector?
	bool const do_classf = ( nIters >= niter_test_conv );

	// ----------------------------------------

	print_message( shown_by_all, "\t<<< NMF-GPU: Non-negative Matrix Factorization on GPU >>>\n"
					"\t\t\t\tSingle-GPU version\n" );

	#if NMFGPU_PROFILING_GLOBAL
		// Total elapsed time
		gettimeofday( &t_tv, NULL );
	#endif

	// ----------------------------------------

	/* Initializes the GPU device.
	 *
	 * In addition:
	 *	- Updates memory_alignment according to the selected GPU device.
	 *	- Updates Kp (i.e., the padded factorization rank).
	 *	- Updates the limits of matrix dimensions.
	 */
	size_t const mem_size = initialize_GPU( gpu_device, K );
	if ( ! mem_size )
		exit(-1);

	// ----------------------------------------

	// Reads input matrix

	struct matrix_tags_t mt = new_empty_matrix_tags();


	//status = init_V( filename, numeric_hdrs, numeric_lbls, input_file_fmt, &mt );
	// if ( status != EXIT_SUCCESS ) {
	// 	shutdown_GPU();
	// 	exit(-1);
	// }
	size_t const nitems = (size_t) nrows * (size_t) pitch;
	bool const wc = true;				// Write-Combined mode
	bool const clear_memory = false;		// Do NOT initialize the allocated memory

	// real *restrict const V = (real *restrict) getHostMemory( nitems * sizeof(real), wc, clear_memory );
	// if ( ! V ) {
	// 	print_error( error_shown_by_all, "Error allocating HOST memory for input matrix.\n" );
	//
	// }
	//
	// // Copies input matrix to the new memory.
	// if ( ! memcpy( V, matrix, nitems * sizeof(real) ) )  {
	// 	print_errnum( sys_error_shown_by_all, errno, "Error initializing input matrix on HOST memory" );
	// 	freeHostMemory( V, "V" );
	// 	exit(-1);
	// }
	// In single-process mode, Vrow and Vcol are just aliases.
	Vcol = Vrow = V;

	// Fails if the factorization rank is too large.
	if ( K > MIN( N, M ) ) {
		print_error( error_shown_by_all, "Error: invalid factorization rank: K=%" PRI_IDX ".\nIt must not be greater "
				"than any of matrix dimensions.\n", K );
		freeHostMemory( Vrow, "V" ); clean_matrix_tags( mt );
		shutdown_GPU();
		exit(-1);
	}

	// ----------------------------------------

	// Setups the GPU device

	status = setup_GPU( mem_size, do_classf );
	if ( status != EXIT_SUCCESS ) {
		freeHostMemory( Vrow, "V" ); clean_matrix_tags( mt );
		shutdown_GPU();
		exit(-1);
	}

	// ----------------------------------------

	// Allocates HOST memory for matrices W and H
	{
		size_t nitems = (size_t) N * (size_t) Kp;
		// W = (real *restrict) getHostMemory( nitems * sizeof(real), false, false );
		// if ( ! W ) {
		// 	print_error( sys_error_shown_by_all, "Error allocating memory for HOST matrix W (N=%" PRI_IDX
		// 			", Kp=%" PRI_IDX ").\n", N, Kp );
		// 	freeHostMemory( Vrow, "V" ); clean_matrix_tags( mt );
		// 	finalize_GPU_device();
		// 	exit(-1);
		// }

		nitems = (size_t) M * (size_t) Kp;
		// H = (real *restrict) getHostMemory( nitems * sizeof(real), false, false );
		// if ( ! H ) {
		// 	print_error( sys_error_shown_by_all, "Error allocating memory for HOST matrix H (M=%" PRI_IDX
		// 			", Kp=%" PRI_IDX ").\n", M, Kp );
		// 	freeHostMemory( W, "W" ); freeHostMemory( Vrow, "V" ); clean_matrix_tags( mt );
		// 	finalize_GPU_device();
		// 	exit(-1);
		// }
	}

	// ----------------------------------------

	// Allocates HOST memory for classification vectors.

	if ( do_classf ) {
		classification = (index_t *restrict) getHostMemory( Mp * sizeof(index_t), false, false );
		if ( ! classification ) {
			print_error( sys_error_shown_by_all, "Error allocating memory for HOST classification vector (M=%" PRI_IDX ", Mp=%"
					PRI_IDX ").\n", M, Mp );
			freeHostMemory( H, "H" ); freeHostMemory( W, "W" ); freeHostMemory( Vrow, "V" ); clean_matrix_tags( mt );
			finalize_GPU_device();
			exit(-1);
		}

		last_classification = (index_t *restrict) getHostMemory( Mp * sizeof(index_t), false, true );	// Initializes with zeros
		if ( ! last_classification ) {
			print_error( sys_error_shown_by_all, "Error allocating memory for HOST classification vector (last, M=%" PRI_IDX
					", Mp=%" PRI_IDX ").\n", M, Mp );
			freeHostMemory(classification, "classification vector"); freeHostMemory( H, "H" ); freeHostMemory( W, "W" );
			freeHostMemory( Vrow, "V" );
			clean_matrix_tags( mt );
			finalize_GPU_device();
			exit(-1);
		}

	} // do_classf

	// ----------------------------------------

	// Executes the NMF Algorithm

	status = nmf( nIters, niter_test_conv, stop_threshold );

	if ( status != EXIT_SUCCESS ) {
		if ( do_classf ) {
			freeHostMemory( last_classification, "previous classification vector" );
			freeHostMemory( classification, "classification vector" );
		}
		freeHostMemory( H, "H" ); freeHostMemory( W, "W" ); freeHostMemory( Vrow, "V" ); clean_matrix_tags( mt );
		finalize_GPU_device();
		exit(-1);
	}

	if ( do_classf ) {
		freeHostMemory( last_classification, "previous classification vector" );
		freeHostMemory( classification, "classification vector" );
	}


	freeHostMemory( H, "H" ); freeHostMemory( W, "W" ); freeHostMemory( Vrow, "V" ); clean_matrix_tags( mt );

	if ( finalize_GPU_device() != EXIT_SUCCESS )
		exit(-1);

	// ----------------------------------------

	print_message( shown_by_all, "Done.\n" );
}
