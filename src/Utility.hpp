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

#ifdef HAVE_MKL
#include <mkl.h>
#else
using MKL_INT = int;
#endif

namespace Utility {
	template<typename T, typename TNorm = double>
	TNorm EuclideanNorm(const std::valarray<T>& x)
	{
		return std::sqrt( std::inner_product(std::begin(x), std::end(x), std::begin(x), TNorm{0}) );
	}

#ifdef HAVE_MKL
	template<typename T>
	inline void MKL_csrgemv(char *transa, MKL_INT *m, T *a, MKL_INT *ia, MKL_INT *ja, T *x, T *y)
	{
		throw std::logic_error("Unknown data type!");
	}

	template<>
	inline void MKL_csrgemv(char *transa, MKL_INT *m, float *a, MKL_INT *ia, MKL_INT *ja, float *x, float *y)
	{
		mkl_scsrgemv(transa, m, a, ia, ja, x, y);
	}

	template<>
	inline void MKL_csrgemv(char *transa, MKL_INT *m, double *a, MKL_INT *ia, MKL_INT *ja, double *x, double *y)
	{
		mkl_dcsrgemv(transa, m, a, ia, ja, x, y);
	}
#endif
}

#endif // !GUARD_UTILITY_HPP
