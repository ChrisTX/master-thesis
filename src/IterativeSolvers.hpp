#ifndef GUARD_ITERATIVE_SOLVERS_HPP
#define GUARD_ITERATIVE_SOLVERS_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <valarray>

#include "CSRMatrix.hpp"
#include "Utility.hpp"

#include <mkl.h>

namespace NavierStokes2D {
	namespace IterativeSolvers {
		void MKL_FGMRES(const Utility::CSRMatrix<double>& A, std::valarray<double>& x, const std::valarray<double>& b, MKL_INT Iterations, MKL_INT IterationsUntilRestart, double ErrorTolerance)
		{
			MKL_INT N = A.GetNumberOfRows();
			assert(x.size() == static_cast<std::size_t>(N));
			MKL_INT RCI_request;
			MKL_INT ipar[128];
			double dpar[128];
			if (!IterationsUntilRestart)
				IterationsUntilRestart = std::min(static_cast<MKL_INT>(150), N);

			std::vector<double> tmp( N * (2 * IterationsUntilRestart + 1) + (IterationsUntilRestart * (IterationsUntilRestart + 9)) / 2 + 1 );
			dfgmres_init(&N, &x[0], const_cast<double*>(&b[0]), &RCI_request, ipar, dpar, tmp.data());
			if (RCI_request != 0)
				throw std::runtime_error("dfmgres_init failed!");
			ipar[4] = Iterations;
			ipar[8] = 1; // do residual stopping test
			ipar[9] = 0; // No user-defined stopping test
			ipar[11] = 1; // Check next-gen vector norm automatically;
			ipar[14] = IterationsUntilRestart;
			dpar[1] = ErrorTolerance;

			dfgmres_check(&N, &x[0], const_cast<double*>(&b[0]), &RCI_request, ipar, dpar, tmp.data());
			if (RCI_request != 0)
				throw std::runtime_error("dfgmres_check failed!");

			while(true) {
				dfgmres(&N, &x[0], const_cast<double*>(&b[0]), &RCI_request, ipar, dpar, tmp.data());

				if (RCI_request)
				{
					if (RCI_request != 1)
						throw std::runtime_error("dfgmres failed!");

					char transa = 'N';
					Utility::MKL_csrgemv(&transa, &N, const_cast<double*>(A.m_Entries.data()), const_cast<MKL_INT*>(A.m_RowIndices.data()), const_cast<MKL_INT*>(A.m_ColumnIndices.data()), &tmp[ipar[21] - 1], &tmp[ipar[22] - 1]);
				}
				else
					break;
			}
			MKL_INT itercount;
			ipar[12] = 0; // Write output in x.
			dfgmres_get(&N, &x[0], const_cast<double*>(&b[0]), &RCI_request, ipar, dpar, tmp.data(), &itercount);

			MKL_Free_Buffers();
		}

		void MKL_PARDISO(Utility::CSRMatrix<double>& A, std::valarray<double>& x, const std::valarray<double>& b)
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
		#ifndef NDEBUG
    		msglvl = 1;
    	#else
    		msglvl = 0;
    	#endif
    		PARDISO(pt.data(), &maxfct, &mnum, &mtype, &phase, &N, A.m_Entries.data(), A.m_RowIndices.data(), A.m_ColumnIndices.data(), &perm_dum, &nrhs, iparm.data(), &msglvl, const_cast<double*>(&b[0]), &x[0], &error);

			if(error != 0)
				throw std::runtime_error("PARDISO failed!");

			double dub_dummy;
			phase = -1;
			PARDISO(pt.data(), &maxfct, &mnum, &mtype, &phase, &N, &dub_dummy, A.m_RowIndices.data(), A.m_ColumnIndices.data(), &perm_dum, &nrhs, iparm.data(), &msglvl, &dub_dummy, &dub_dummy, &error);
		}
	}
}

#endif // !GUARD_ITERATIVE_SOLVERS_HPP
