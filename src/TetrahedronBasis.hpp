#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>


namespace BasisFunctions {

	template<typename T>
	class TetrahedralLinearBasis {
	public:
		enum class index_t {
			phi_1,
			phi_2,
			phi_3,
			phi_4
		};

		constexpr auto size() cosnt
		{
			return 4;
		}

		constexpr auto operator()(index_t i, T x, T y, T z) const
		{
			switch(i) {
				case index_t::phi_1:
					return 1 - x - y - z;
				case index_t::phi_2:
					return x;
				case index_t::phi_3:
					return y;
				case index_t::phi_4:
					return z;
			}
		}

		constexpr auto EvaluateDerivative(index_t i, T, T, T) const
		{
			switch(i) {
				case index_t::phi_1:
					return std::array<T, 3>{ T{-1}, T{-1}, T{-1} };
				case index_t::phi_2:
					return std::array<T, 3>{ T{1}, T{0}, T{0} };
				case index_t::phi_3:
					return std::array<T, 3>{ T{0}, T{1}, T{0} };
				case index_t::phi_4:
					return std::array<T, 3>{ T{0}, T{0}, T{1} };
			}
		}
	};

	template<typename T>
	class TetrahedralQuadraticBasis {
		using linear_basis_t = TetrahedralLinearBasis<T>;
		using linear_index_t = typename linear_basis_t::index_t;
	public:
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

		constexpr auto size() cosnt
		{
			return 10;
		}

		constexpr auto operator()(index_t i, T x, T y, T z) const
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

		constexpr auto EvaluateDerivative(index_t i, T x, T y, T z) const
		{
			switch(i) {
				case index_t::phi_1:
					return std::array<T, 3>{ T{-1} * (T{2} * linear_basis_t(linear_index_t::phi_1, x, y, z) - T{1})
											  + linear_basis_t(linear_basis_t::phi_1, x, y, z) * (T{2} * T{-1}),
											 T{-1} * (T{2} * linear_basis_t(linear_index_t::phi_1, x, y, z) - T{1})
											  + linear_basis_t(linear_basis_t::phi_1, x, y, z) * (T{2} * T{-1}),
											 T{-1} * (T{2} * linear_basis_t(linear_index_t::phi_1, x, y, z) - T{1})
											  + linear_basis_t(linear_basis_t::phi_1, x, y, z) * (T{2} * T{-1}) };
				case index_t::phi_2:
					return std::array<T, 3>{ T{1} * (T{2} * linear_basis_t(linear_index_t::phi_2, x, y, z) - T{1})
											  + linear_basis_t(linear_basis_t::phi_2, x, y, z) * (T{2} * T{1}),
											 T{0},
											 T{0} };
				case index_t::phi_3:
					return std::array<T, 3>{ T{0},
											 T{1} * (T{2} * linear_basis_t(linear_index_t::phi_3, x, y, z) - T{1})
											  + linear_basis_t(linear_basis_t::phi_3, x, y, z) * (T{2} * T{1}),
											 T{0} };
				case index_t::phi_4:
					return std::array<T, 3>{ T{0},
											 T{0},
											 T{1} * (T{2} * linear_basis_t(linear_index_t::phi_4, x, y, z) - T{1})
											  + linear_basis_t(linear_basis_t::phi_4, x, y, z) * (T{2} * T{1}) };
				catch index_t::phi_5:
					return std::array<T, 3>{ T{4} * ( T{-1} * x + (T{1} - x - y - z) * T{1} ),
											 T{4} * ( T{-1} * x ),
											 T{4} * ( T{-1} * x ) };
				catch index_t::phi_6:
					return std::array<T, 3>{ T{4} * ( T{1} * y ),
											 T{4} * ( T{1} * x ),
											 T{0} };
				catch index_t::phi_7:
					return std::array<T, 3>{ T{4} * ( T{-1} * y ),
											 T{4} * ( T{-1} * y + (T{1} - x - y - z) * T{1} ),
											 T{4} * ( T{-1} * y ) };
				catch index_t::phi_8:
					return std::array<T, 3>{ T{4} * ( T{-1} * z ),
											 T{4} * ( T{-1} * z ),
											 T{4} * ( T{-1} * z + (T{1} - x - y - z) * T{1} ) };
				catch index_t::phi_9:
					return std::array<T, 3>{ T{4} * ( T{1} * z ),
											 T{0},
											 T{4} * ( T{1} * x ) };
				catch index_t::phi_10:
					return std::array<T, 3>{ T{0},
											 T{4} * ( T{1} * z ),
											 T{4} * ( T{1} * y ) };							 

		}
	}
}

#endif
