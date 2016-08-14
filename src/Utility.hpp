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

namespace NavierStokes2D {
	namespace Utility {
		template<typename T, typename TNorm = double>
		TNorm EuclideanNorm(const std::valarray<T>& x)
		{
			return std::sqrt( std::inner_product(std::begin(x), std::end(x), std::begin(x), TNorm{0}) );
		}

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

		template<typename T>
		constexpr const char* TypeVTKName()
		{
			// This form should never be used.
			return nullptr;
		}

		template<>
		constexpr const char* TypeVTKName<float>()
		{
			// This is by no means guaranteed to be a human readable name!
			return "float";
		}

		template<>
		constexpr const char* TypeVTKName<double>()
		{
			// This is by no means guaranteed to be a human readable name!
			return "double";
		}

		template<typename T>
		class TicketMap {
		public:
			using size_type = std::size_t;

			TicketMap() : m_TicketCounter{ 0 } {}

			inline void insert(T new_key)
			{
				const auto find_it = m_TicketMap.find(new_key);
				if (find_it == m_TicketMap.end())
					m_TicketMap.insert(std::make_pair(new_key, m_TicketCounter++));
			}

			inline size_type lookup(const T& key_to_lookup) const
			{
				const auto find_it = m_TicketMap.find(key_to_lookup);
				assert(find_it != m_TicketMap.end());

				return find_it->second;
			}

			inline bool contains(const T& key_to_lookup, size_type& ticket_value) const
			{
				const auto find_it = m_TicketMap.find(key_to_lookup);
				if (find_it == m_TicketMap.end()) {
					ticket_value = find_it->second;
					return true;
				}
				else {
					return false;
				}
			}

			inline std::size_t size() const
			{
				return m_TicketMap.size();
			}

		protected:
			size_type m_TicketCounter;
			std::unordered_map<T, size_type> m_TicketMap;
		};
	}
}

#endif // !GUARD_UTILITY_HPP
