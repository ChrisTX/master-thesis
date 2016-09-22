#ifndef TETRAHEDRAL_BASIS_HPP
#define TETRAHEDRAL_BASIS_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#ifdef HAVE_ECONSTEXPR
#define ECONSTEXPR constexpr
#else
#define ECONSTEXPR
#endif

namespace BasisFunctions {

	template<typename T>
	class TetrahedralLinearBasis {
	public:
		using point_t = std::array<T, 3>;

		enum class index_t {
			phi_1,
			phi_2,
			phi_3,
			phi_4
		};

		constexpr auto size() const
		{
			return 4;
		}

		ECONSTEXPR auto operator()(const index_t i, const point_t& p) const
		{
			switch(i) {
				case index_t::phi_1:
					return 1 - p[0] - p[1] - p[2];
				case index_t::phi_2:
					return p[0];
				case index_t::phi_3:
					return p[1];
				case index_t::phi_4:
					return p[2];
			}

			assert(false);
			return std::move(std::numeric_limits<T>::signaling_NaN());
		}

		ECONSTEXPR auto EvaluateDerivative(const index_t i, const point_t&) const
		{
			switch(i) {
				case index_t::phi_1:
					return std::move(std::array<T, 3>{ T{-1}, T{-1}, T{-1} });
				case index_t::phi_2:
					return std::move(std::array<T, 3>{ T{1}, T{0}, T{0} });
				case index_t::phi_3:
					return std::move(std::array<T, 3>{ T{0}, T{1}, T{0} });
				case index_t::phi_4:
					return std::move(std::array<T, 3>{ T{0}, T{0}, T{1} });
			}

			assert(false);
			return std::move(std::array<T, 3>{std::numeric_limits<T>::signaling_NaN(), std::numeric_limits<T>::signaling_NaN(), std::numeric_limits<T>::signaling_NaN()});
		}
	};

	template<typename T>
	class TetrahedralQuadraticBasis {
		using linear_basis_t = TetrahedralLinearBasis<T>;
		using linear_index_t = typename linear_basis_t::index_t;
	public:
		using point_t = std::array<T, 3>;

		enum class index_t {
			phi_1,
			phi_2,
			phi_3,
			phi_4,
			phi_5,
			phi_6,
			phi_7,
			phi_8,
			phi_9,
			phi_10
		};

		constexpr auto size() const
		{
			return 10;
		}

		ECONSTEXPR auto operator()(const index_t i, const point_t&) const
		{
			switch(i) {
				case index_t::phi_1:
					return linear_basis_t(linear_index_t::phi_1) * ( T{2} * linear_basis_t(linear_index_t::phi_1) - 1);
				case index_t::phi_2:
					return linear_basis_t(linear_index_t::phi_2) * ( T{2} * linear_basis_t(linear_index_t::phi_2) - 1);
				case index_t::phi_3:
					return linear_basis_t(linear_index_t::phi_3) * ( T{2} * linear_basis_t(linear_index_t::phi_3) - 1);
				case index_t::phi_4:
					return linear_basis_t(linear_index_t::phi_4) * ( T{2} * linear_basis_t(linear_index_t::phi_4) - 1);
				case index_t::phi_5:
					return T{4} * linear_basis_t(linear_index_t::phi_1) * linear_basis_t(linear_index_t::phi_2);
				case index_t::phi_6:
					return T{4} * linear_basis_t(linear_index_t::phi_2) * linear_basis_t(linear_index_t::phi_3);
				case index_t::phi_7:
					return T{4} * linear_basis_t(linear_index_t::phi_1) * linear_basis_t(linear_index_t::phi_3);
				case index_t::phi_8:
					return T{4} * linear_basis_t(linear_index_t::phi_1) * linear_basis_t(linear_index_t::phi_4);
				case index_t::phi_9:
					return T{4} * linear_basis_t(linear_index_t::phi_2) * linear_basis_t(linear_index_t::phi_4);
				case index_t::phi_10:
					return T{4} * linear_basis_t(linear_index_t::phi_3) * linear_basis_t(linear_index_t::phi_4);
			}
		}

		ECONSTEXPR auto EvaluateDerivative(const index_t i, const point_t& p) const
		{
			switch(i) {
				case index_t::phi_1:
					return std::array<T, 3>{ T{-1} * (T{2} * linear_basis_t(linear_index_t::phi_1, p) - T{1})
											  + linear_basis_t(linear_basis_t::phi_1, p) * (T{2} * T{-1}),
											 T{-1} * (T{2} * linear_basis_t(linear_index_t::phi_1, p) - T{1})
											  + linear_basis_t(linear_basis_t::phi_1, p) * (T{2} * T{-1}),
											 T{-1} * (T{2} * linear_basis_t(linear_index_t::phi_1, p) - T{1})
											  + linear_basis_t(linear_basis_t::phi_1, p) * (T{2} * T{-1}) };
				case index_t::phi_2:
					return std::array<T, 3>{ T{1} * (T{2} * linear_basis_t(linear_index_t::phi_2, p) - T{1})
											  + linear_basis_t(linear_basis_t::phi_2, p) * (T{2} * T{1}),
											 T{0},
											 T{0} };
				case index_t::phi_3:
					return std::array<T, 3>{ T{0},
											 T{1} * (T{2} * linear_basis_t(linear_index_t::phi_3, p) - T{1})
											  + linear_basis_t(linear_basis_t::phi_3, p) * (T{2} * T{1}),
											 T{0} };
				case index_t::phi_4:
					return std::array<T, 3>{ T{0},
											 T{0},
											 T{1} * (T{2} * linear_basis_t(linear_index_t::phi_4, p) - T{1})
											  + linear_basis_t(linear_basis_t::phi_4, p) * (T{2} * T{1}) };
				case index_t::phi_5:
					return std::array<T, 3>{ T{4} * ( T{-1} * linear_basis_t(linear_index_t::phi_2, p) + linear_basis_t(linear_index_t::phi_1, p) * T{1} ),
											 T{4} * ( T{-1} * linear_basis_t(linear_index_t::phi_2, p) ),
											 T{4} * ( T{-1} * linear_basis_t(linear_index_t::phi_2, p) ) };
				case index_t::phi_6:
					return std::array<T, 3>{ T{4} * ( T{1} * linear_basis_t(linear_index_t::phi_3, p) ),
											 T{4} * ( T{1} * linear_basis_t(linear_index_t::phi_2, p) ),
											 T{0} };
				case index_t::phi_7:
					return std::array<T, 3>{ T{4} * ( T{-1} * linear_basis_t(linear_index_t::phi_3, p) ),
											 T{4} * ( T{-1} * linear_basis_t(linear_index_t::phi_3, p) + linear_basis_t(linear_index_t::phi_1, p) * T{1} ),
											 T{4} * ( T{-1} * linear_basis_t(linear_index_t::phi_3, p) ) };
				case index_t::phi_8:
					return std::array<T, 3>{ T{4} * ( T{-1} * linear_basis_t(linear_index_t::phi_4, p) ),
											 T{4} * ( T{-1} * linear_basis_t(linear_index_t::phi_4, p) ),
											 T{4} * ( T{-1} * linear_basis_t(linear_index_t::phi_4, p) + linear_basis_t(linear_index_t::phi_1, p) * T{1} ) };
				case index_t::phi_9:
					return std::array<T, 3>{ T{4} * ( T{1} * linear_basis_t(linear_index_t::phi_4, p) ),
											 T{0},
											 T{4} * ( T{1} * linear_basis_t(linear_index_t::phi_2, p) ) };
				case index_t::phi_10:
					return std::array<T, 3>{ T{0},
											 T{4} * ( T{1} * linear_basis_t(linear_index_t::phi_4, p) ),
											 T{4} * ( T{1} * linear_basis_t(linear_index_t::phi_3, p) ) };							 
			}
		}
	};
}

#endif
