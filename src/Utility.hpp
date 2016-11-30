#ifndef GUARD_UTILITY_HPP
#define GUARD_UTILITY_HPP

#include <cmath>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <valarray>

#include <mkl.h>

namespace Utility {
	template<typename T, typename TNorm = double>
	TNorm EuclideanNorm(const std::valarray<T>& x)
	{
		return std::sqrt( std::inner_product(std::begin(x), std::end(x), std::begin(x), TNorm{0}) );
	}

	inline void MKL_csrgemv(char *transa, MKL_INT *m, float *a, MKL_INT *ia, MKL_INT *ja, float *x, float *y)
	{
		mkl_scsrgemv(transa, m, a, ia, ja, x, y);
	}

	inline void MKL_csrgemv(char *transa, MKL_INT *m, double *a, MKL_INT *ia, MKL_INT *ja, double *x, double *y)
	{
		mkl_dcsrgemv(transa, m, a, ia, ja, x, y);
	}

	inline void MKL_csrsymv(char *uplo, MKL_INT *m, float *a, MKL_INT *ia, MKL_INT *ja, float *x, float *y)
	{
		mkl_scsrsymv(uplo, m, a, ia, ja, x, y);
	}

	inline void MKL_csrsymv(char *uplo, MKL_INT *m, double *a, MKL_INT *ia, MKL_INT *ja, double *x, double *y)
	{
		mkl_dcsrsymv(uplo, m, a, ia, ja, x, y);
	}
}

#endif // !GUARD_UTILITY_HPP
