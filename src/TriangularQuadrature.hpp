#ifndef TRIANGULAR_QUADRATURE_HPP
#define TRIANGULAR_QUADRATURE_HPP

#include <type_traits>
#include <utility>
#include <array>

#include "QuadratureFormulas.hpp"

namespace QuadratureFormulas {

	namespace Triangles {

		template<typename T>
		struct Formula_2DD1 {
			using point_t = std::array<T, 2>;
			const std::array<T, 1> weights{ T{1}/T{2} };
			const std::array<point_t, 1> points{ { T{1}/T{3}, T{1}/T{3} } };

			template<typename F>
			auto operator()(const F& f_integrand) const {
				return EvaluateQuadrature(points, weights, f_integrand);
			}
		};

		template<typename T>
		struct Formula_2DD2 {
			using point_t = std::array<T, 2>;
			const std::array<T, 3> weights{ T{1}/T{6}, T{1}/T{6}, T{1}/T{6} };
			const std::array<point_t, 3> points{ point_t{ T{0}, T{0} }, point_t{ T{1}, T{0} }, point_t{ T{0}, T{1} } };

			template<typename F>
			auto operator()(const F& f_integrand) const {
				return EvaluateQuadrature(points, weights, f_integrand);
			}
		};

		template<typename T>
		struct Formula_2DD3 {
			using point_t = std::array<T, 2>;
			const std::array<T, 3> weights{ T{1}/T{6}, T{1}/T{6}, T{1}/T{6} };
			const std::array<point_t, 3> points{ point_t{ T{1}/T{2}, T{0} }, point_t{ T{1}/T{2}, T{1}/T{2} }, point_t{ T{0}, T{1}/T{2} } };

			template<typename F>
			auto operator()(const F& f_integrand) const {
				return EvaluateQuadrature(points, weights, f_integrand);
			}
		};

		template<typename T>
		struct Formula_2DD4 {
			using point_t = std::array<T, 2>;
			const std::array<T, 3> weights{ T{1}/T{6}, T{1}/T{6}, T{1}/T{6} };
			const std::array<point_t, 3> points{ point_t{ T{1}/T{6}, T{1}/T{6} }, point_t{ T{4}/T{6}, T{1}/T{6} }, point_t{ T{1}/T{6}, T{4}/T{6} } };

			template<typename F>
			auto operator()(const F& f_integrand) const {
				return EvaluateQuadrature(points, weights, f_integrand);
			}
		};


		template<typename T>
		struct Formula_2DD5 {
			using point_t = std::array<T, 2>;
			const std::array<T, 7> weights{ T{3}/T{120}, T{3}/T{120}, T{3}/T{120}, T{8}/T{120}, T{8}/T{120}, T{8}/T{120}, T{27}/T{120} };
			const std::array<point_t, 7> points{ point_t{ T{0}, T{0} }, point_t{ T{1}, T{0} }, point_t{ T{0}, T{1} }, point_t{ T{1}/T{2}, T{0} }, point_t{ T{1}/T{2}, T{1}/T{2} }, point_t{ T{0}, T{1}/T{2} }, point_t{ T{1}/T{3}, T{1}/T{3} } };

			template<typename F>
			auto operator()(const F& f_integrand) const {
				return EvaluateQuadrature(points, weights, f_integrand);
			}
		};


		template<typename T>
		struct Formula_2DD6 {
		private:
			constexpr static T sqrt_of_fifteen = T{3.8729833462074168851792653997824};
			constexpr static T vma = (T{6} - sqrt_of_fifteen)/T{21};
			constexpr static T vpa = (T{6} + sqrt_of_fifteen)/T{21};
			constexpr static T vmb = (T{9} - T{2} * sqrt_of_fifteen)/T{21};
			constexpr static T vpb = (T{9} + T{2} * sqrt_of_fifteen)/T{21};
			constexpr static T wa = (T{155} - sqrt_of_fifteen)/T{2400};
			constexpr static T wb = (T{155} + sqrt_of_fifteen)/T{2400};
		public:
			using point_t = std::array<T, 2>;
			const std::array<T, 7> weights{ wa, wa, wa, wb, wb, wb, T{9}/T{80} };
			const std::array<point_t, 7> points{
				point_t{ vma, vma }, point_t{ vpb, vma }, point_t{ vma, vpb }, point_t{ vpa, vmb }, point_t{ vpa, vpa }, point_t{ vmb, vpa }, point_t{ T{1}/T{3}, T{1}/T{3} }
			};

			template<typename F>
			auto operator()(const F& f_integrand) const {
				return EvaluateQuadrature(points, weights, f_integrand);
			}
		};

		template<typename T>
		class ReferenceTransform {
		public:
			using point_t = std::array<T, 2>;
			using surf_point_t = std::array<T, 3>;
			using triangle_t = std::array<surf_point_t, 3>;
		protected:
			using size_type_t = typename triangle_t::size_type;
			std::array<surf_point_t, 2> m_TransformMatrix;
			surf_point_t m_a;
			T m_det;

			void CalculateGramDeterminant() {
				const auto det11 = m_TransformMatrix[0][0] * m_TransformMatrix[0][0] + m_TransformMatrix[0][1] * m_TransformMatrix[0][1] + m_TransformMatrix[0][2] * m_TransformMatrix[0][2];
				const auto det12 = m_TransformMatrix[0][0] * m_TransformMatrix[1][0] + m_TransformMatrix[0][1] * m_TransformMatrix[1][1] + m_TransformMatrix[0][2] * m_TransformMatrix[1][2];
				const auto det21 = det12;
				const auto det22 = m_TransformMatrix[1][0] * m_TransformMatrix[1][0] + m_TransformMatrix[1][1] * m_TransformMatrix[1][1] + m_TransformMatrix[1][2] * m_TransformMatrix[1][2];
				m_det = std::sqrt( det11 * det22 - det21 * det12 );
			}

		public:
			ReferenceTransform(const triangle_t& target_element) {
				m_a = target_element[0];
				for(auto i = size_type_t{0}; i < m_TransformMatrix.size(); ++i)
					m_TransformMatrix[i] = target_element[i + 1];

				for(auto i = size_type_t{0}; i + 1 < target_element.size(); ++i)
					for(auto j = size_type_t{0}; j < size_type_t{3}; ++j)
						m_TransformMatrix[i][j] -= m_a[j];
				
				CalculateGramDeterminant();
			}

			auto GetDeterminantSqrt() const {
				assert(std::isfinite(m_det) && m_det > 5 * std::numeric_limits<T>::epsilon());
				return m_det;
			}

			auto operator()(const point_t& x0) const {
				auto xt = surf_point_t{};
				for (auto i = size_type_t{0}; i < m_TransformMatrix.size(); ++i)
					for (auto j = size_type_t{0}; j < size_type_t{3}; ++j)
						xt[j] += m_TransformMatrix[i][j] * x0[i];
				
				for(auto j = size_type_t{0}; j < size_type_t{3}; ++j)
					xt[j] += m_a[j];

				return xt;
			}
		};

		template<typename T>
		class ReferenceTransform2D {
		public:
			using point_t = std::array<T, 2>;
			using triangle_t = std::array<point_t, 3>;
		protected:
			using size_type_t = typename triangle_t::size_type;
			std::array<point_t, 2> m_TransformMatrix;
			point_t m_a;
			T m_det;

			void CalculateDeterminant() {
				const auto det11 = m_TransformMatrix[0][0];
				const auto det12 = m_TransformMatrix[1][0];
				const auto det21 = m_TransformMatrix[0][1];
				const auto det22 = m_TransformMatrix[1][1];
				m_det = std::abs( det11 * det22 - det21 * det12 );
			}

		public:
			ReferenceTransform2D(const triangle_t& target_element) {
				m_a = target_element[0];
				for(auto i = size_type_t{0}; i < m_TransformMatrix.size(); ++i)
					m_TransformMatrix[i] = target_element[i + 1];

				for(auto i = size_type_t{0}; i + 1 < target_element.size(); ++i)
					for(auto j = size_type_t{0}; j < size_type_t{2}; ++j)
						m_TransformMatrix[i][j] -= m_a[j];
				
				CalculateDeterminant();
			}

			auto GetDeterminantAbs() const {
				assert(std::isfinite(m_det) && std::abs(m_det) > 5 * std::numeric_limits<T>::epsilon());
				return m_det;
			}

			auto operator()(const point_t& x0) const {
				auto xt = point_t{};
				for(auto i = size_type_t{0}; i < m_TransformMatrix.size(); ++i)
					for(auto j = size_type_t{0}; j < size_type_t{2}; ++j)
						xt[j] += m_TransformMatrix[i][j] * x0[i];
				
				for(auto j = size_type_t{0}; j < size_type_t{2}; ++j)
					xt[j] += m_a[j];

				return xt;
			}
		};
	}
}

#endif
