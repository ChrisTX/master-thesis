#ifndef GUARD_ITERATIVE_SOLVERS_HPP
#define GUARD_ITERATIVE_SOLVERS_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "CSRMatrix.hpp"
#include "Utility.hpp"

#include <mkl.h>

namespace IterativeSolvers {
	void MKL_FGMRES(const Utility::CSRMatrix<double>& A, std::vector<double>& x, const std::vector<double>& b, MKL_INT Iterations, MKL_INT IterationsUntilRestart, double ErrorTolerance)
	{
		MKL_INT N = A.GetNumberOfRows();
		assert(x.size() == static_cast<std::size_t>(N));
		MKL_INT RCI_request;
		std::array<MKL_INT, 128> ipar;
		std::array<double, 128> dpar;

		if (!Iterations)
			Iterations = std::min(static_cast<MKL_INT>(150), N);
		if (!IterationsUntilRestart)
			IterationsUntilRestart = std::min(static_cast<MKL_INT>(150), N);

		std::vector<double> tmp( N * (2 * IterationsUntilRestart + 1) + (IterationsUntilRestart * (IterationsUntilRestart + 9)) / 2 + 1 );
		dfgmres_init(&N, &x[0], const_cast<double*>(&b[0]), &RCI_request, ipar.data(), dpar.data(), tmp.data());
		if (RCI_request != 0)
			throw std::runtime_error("dfmgres_init failed!");
		ipar[1] = 6;
		ipar[5] = 1;
		ipar[6] = 1;
		ipar[4] = Iterations;
		ipar[9] = 0; // No user-defined stopping test
		ipar[11] = 1; // Check next-gen vector norm automatically;
		ipar[14] = IterationsUntilRestart;
		if (ErrorTolerance > 0) {
			ipar[8] = 1; // do residual stopping test
			dpar[1] = ErrorTolerance;
		}

		dfgmres_check(&N, &x[0], const_cast<double*>(&b[0]), &RCI_request, ipar.data(), dpar.data(), tmp.data());
		if (RCI_request != 0)
			throw std::runtime_error("dfgmres_check failed!");

		while(true) {
			dfgmres(&N, &x[0], const_cast<double*>(&b[0]), &RCI_request, ipar.data(), dpar.data(), tmp.data());

			if (RCI_request)
			{
				if (RCI_request != 1)
					throw std::runtime_error("dfgmres failed!");

#ifdef SYMMETRIC_ASSEMBLY
				char uplo = 'U';
				Utility::MKL_csrsymv(&uplo, &N, const_cast<double*>(A.m_Entries.data()), const_cast<MKL_INT*>(A.m_RowIndices.data()), const_cast<MKL_INT*>(A.m_ColumnIndices.data()), &tmp[ipar[21] - 1], &tmp[ipar[22] - 1]);
#else
				char transa = 'N';
				Utility::MKL_csrgemv(&transa, &N, const_cast<double*>(A.m_Entries.data()), const_cast<MKL_INT*>(A.m_RowIndices.data()), const_cast<MKL_INT*>(A.m_ColumnIndices.data()), &tmp[ipar[21] - 1], &tmp[ipar[22] - 1]);
#endif
			}
			else
				break;
		}
		MKL_INT itercount;
		ipar[12] = 0; // Write output in x.
		dfgmres_get(&N, &x[0], const_cast<double*>(&b[0]), &RCI_request, ipar.data(), dpar.data(), tmp.data(), &itercount);

		MKL_Free_Buffers();
	}

	void MKL_PARDISO(const Utility::CSRMatrix<double>& A, std::vector<double>& x, const std::vector<double>& b)
	{
		std::array<void *, 64> pt{};
		std::array<MKL_INT, 64> iparm{};
		MKL_INT maxfct, mnum, phase, error, msglvl, nrhs, mtype, N, perm_dum;
		iparm[1] = 2; // Nested dissection from METIS
		maxfct = 1; // Maximally one factorization.
		mnum = 1; // Use first (and only) factorization.
		mtype = 11; // real and nonsymmetric
		phase = 13; // Analysis, numerical factorization, solve, iterative refinement
		N = A.GetNumberOfRows(); // Number of rows
		assert(x.size() == static_cast<std::size_t>(N));
		nrhs = 1;

		for (auto i = 0; i < iparm.size(); ++i)
			iparm[i] = 0;

		iparm[0] = 1;         /* No solver default */
		iparm[1] = 2;         /* Fill-in reordering from METIS */
		iparm[3] = 0;         /* No iterative-direct algorithm */
		iparm[4] = 0;         /* No user fill-in reducing permutation */
		iparm[5] = 0;         /* Write solution into x */
		iparm[6] = 0;         /* Not in use */
		iparm[7] = 0;         /* Max numbers of iterative refinement steps */
		iparm[8] = 0;         /* Not in use */
		iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
		iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
		iparm[11] = 0;        /* Not in use */
		iparm[12] = 1;        /* Maximum weighted matching algorithm is switched-off */
							  /* (default for symmetric). Try iparm[12] = 1 in case of inappropriate accuracy */
		iparm[13] = 0;        /* Output: Number of perturbed pivots */
		iparm[14] = 0;        /* Not in use */
		iparm[15] = 0;        /* Not in use */
		iparm[16] = 0;        /* Not in use */
		iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
		iparm[18] = -1;       /* Output: Mflops for LU factorization */
		iparm[19] = 0;        /* Output: Numbers of CG Iterations */

		msglvl = 1;
	#ifndef NDEBUG
		iparm[26] = 1;
	#endif
		PARDISO(pt.data(), &maxfct, &mnum, &mtype, &phase, &N, A.m_Entries.data(), A.m_RowIndices.data(), A.m_ColumnIndices.data(), &perm_dum, &nrhs, iparm.data(), &msglvl, const_cast<double*>( b.data() ), x.data(), &error);

		if(error != 0)
			throw std::runtime_error("PARDISO failed!");

		double dub_dummy;
		phase = -1;
		PARDISO(pt.data(), &maxfct, &mnum, &mtype, &phase, &N, &dub_dummy, A.m_RowIndices.data(), A.m_ColumnIndices.data(), &perm_dum, &nrhs, iparm.data(), &msglvl, &dub_dummy, &dub_dummy, &error);
	}

	void MKL_PARDISO_SYM(const Utility::CSRMatrix<double>& A, std::vector<double>& x, const std::vector<double>& b)
	{
		std::array<void *, 64> pt{};
		std::array<MKL_INT, 64> iparm{};
		MKL_INT maxfct, mnum, phase, error, msglvl, nrhs, mtype, N, perm_dum;
		iparm[1] = 2; // Nested dissection from METIS
		maxfct = 1; // Maximally one factorization.
		mnum = 1; // Use first (and only) factorization.
		mtype = -2; // real and symmetric
		phase = 13; // Analysis, numerical factorization, solve, iterative refinement
		N = A.GetNumberOfRows(); // Number of rows
		assert(x.size() == static_cast<std::size_t>(N));
		nrhs = 1;

		for (auto i = 0; i < iparm.size(); ++i)
			iparm[i] = 0;

		iparm[0] = 1;         /* No solver default */
		iparm[1] = 2;         /* Fill-in reordering from METIS */
		iparm[3] = 0;         /* No iterative-direct algorithm */
		iparm[4] = 0;         /* No user fill-in reducing permutation */
		iparm[5] = 0;         /* Write solution into x */
		iparm[6] = 0;         /* Not in use */
		iparm[7] = 0;         /* Max numbers of iterative refinement steps */
		iparm[8] = 0;         /* Not in use */
		iparm[9] = 8;        /* Perturb the pivot elements with 1E-13 */
		iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
		iparm[11] = 0;        /* Not in use */
		iparm[12] = 1;        /* Maximum weighted matching algorithm is switched-off */
							  /* (default for symmetric). Try iparm[12] = 1 in case of inappropriate accuracy */
		iparm[13] = 1;        /* Output: Number of perturbed pivots */
		iparm[14] = 0;        /* Not in use */
		iparm[15] = 0;        /* Not in use */
		iparm[16] = 0;        /* Not in use */
		iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
		iparm[18] = -1;       /* Output: Mflops for LU factorization */
		iparm[19] = 0;        /* Output: Numbers of CG Iterations */

		msglvl = 1;
#ifndef NDEBUG
		iparm[26] = 1;
#endif
		PARDISO(pt.data(), &maxfct, &mnum, &mtype, &phase, &N, A.m_Entries.data(), A.m_RowIndices.data(), A.m_ColumnIndices.data(), &perm_dum, &nrhs, iparm.data(), &msglvl, const_cast<double*>(b.data()), x.data(), &error);

		if (error != 0)
			throw std::runtime_error("PARDISO failed!");

		double dub_dummy;
		phase = -1;
		PARDISO(pt.data(), &maxfct, &mnum, &mtype, &phase, &N, &dub_dummy, A.m_RowIndices.data(), A.m_ColumnIndices.data(), &perm_dum, &nrhs, iparm.data(), &msglvl, &dub_dummy, &dub_dummy, &error);
	}
}

#endif // !GUARD_ITERATIVE_SOLVERS_HPP
