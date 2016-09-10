#ifndef TETRAHEDRAL_QUADRATURE_HPP
#define TETRAHEDRAL_QUADRATURE_HPP

#include <type_traits>
#include <utility>
#include <array>

#include "QuadratureFormulas.hpp"

namespace QuadratureFormulas {

	namespace Tetrahedra {

		template<typename T>
		struct Formula_3DT1 {
			using point_t = std::array<T, 3>;
			const std::array<T, 1> weights{ T{1}/T{6} };
			const std::array<point_t, 1> points{ { T{1}/T{4}, T{1}/T{4}, T{1}/T{4} } };

			template<typename F>
			auto operator()(const F& f_integrand) const {
				return EvaluateQuadrature(points, weights, f_integrand);
			}
		};

		template<typename T>
		struct Formula_3DT2 {
			using point_t = std::array<T, 3>;
			const std::array<T, 4> weights{ T{1}/T{24}, T{1}/T{24}, T{1}/T{24}, T{1}/T{24} };
			const std::array<point_t, 4> points{ { T{0}, T{0}, T{0} }, { T{1}, T{0}, T{0} }, { T{0}, T{1}, T{0} }, { T{0}, T{0}, T{1} } };

			template<typename F>
			auto operator()(const F& f_integrand) const {
				return EvaluateQuadrature(points, weights, f_integrand);
			}
		};

		template<typename T>
		struct Formula_3DT3 {
		private:
			constexpr static T sqrt_of_five = T{2.2360679774997896964091736687313};
		public:
			using point_t = std::array<T, 3>;
			const std::array<T, 4> weights{ T{1}/T{24}, T{1}/T{24}, T{1}/T{24}, T{1}/T{24} };
			const std::array<point_t, 4> points{
				{ (T{5} - sqrt_of_five)/T{20}, (T{5} - sqrt_of_five)/T{20}, (T{5} - sqrt_of_five)/T{20} }, 
				{ (T{5} + T{3} * sqrt_of_five)/T{20}, (T{5} - sqrt_of_five)/T{20}, (T{5} - sqrt_of_five)/T{20} }, 
				{ (T{5} - sqrt_of_five)/T{20}, (T{5} + T{3} * sqrt_of_five)/T{20}, (T{5} - sqrt_of_five)/T{20} }, 
				{ (T{5} - sqrt_of_five)/T{20}, (T{5} - sqrt_of_five)/T{20}, (T{5} + T{3} * sqrt_of_five)/T{20} } 
			};

			template<typename F>
			auto operator()(const F& f_integrand) const {
				return EvaluateQuadrature(points, weights, f_integrand);
			}
		};

		template<typename T>
		class ReferenceTransform {
		public:
			using point_t = std::array<T, 3>;
			using tetrahedron_t = std::array<point_t, 4>;

		protected:
			using size_type_t = typename tetrahedron_t::size_type;
			std::array<point_t, 3> m_TransformMatrix;
			std::array<point_t, 3> m_InverseTransformMatrix;
			point_t m_d;
			T m_det;

			void CalculateDeterminant() {
				m_det = T{0};
				m_det += m_TransformMatrix[0][0] * (m_TransformMatrix[1][1] * m_TransformMatrix[2][2] - m_TransformMatrix[2][1] * m_TransformMatrix[1][2]);
				m_det -= m_TransformMatrix[1][0] * (m_TransformMatrix[0][1] * m_TransformMatrix[2][2] - m_TransformMatrix[2][1] * m_TransformMatrix[0][2]);
				m_det += m_TransformMatrix[2][0] * (m_TransformMatrix[0][1] * m_TransformMatrix[1][2] - m_TransformMatrix[1][1] * m_TransformMatrix[0][2]); 
			}

			void CalculateInverse() {
				const auto& a = m_TransformMatrix[0][0];
				const auto& b = m_TransformMatrix[1][0];
				const auto& c = m_TransformMatrix[2][0];
				const auto& d = m_TransformMatrix[0][1];
				const auto& e = m_TransformMatrix[1][1];
				const auto& f = m_TransformMatrix[2][1];
				const auto& g = m_TransformMatrix[0][2];
				const auto& h = m_TransformMatrix[1][2];
				const auto& i = m_TransformMatrix[2][2];

				m_InverseTransformMatrix[0][0] = 			( e * i - f * h ) / m_det;
				m_InverseTransformMatrix[1][0] = T{-1} * 	( d * i - f * g ) / m_det;
				m_InverseTransformMatrix[2][0] = 			( d * h - e * g ) / m_det;
				m_InverseTransformMatrix[0][1] = T{-1} * 	( b * i - c * h ) / m_det;
				m_InverseTransformMatrix[1][1] = 			( a * i - c * g ) / m_det;
				m_InverseTransformMatrix[2][1] = T{-1} * 	( a * h - b * g ) / m_det;
				m_InverseTransformMatrix[0][2] = 			( b * f - c * e ) / m_det;
				m_InverseTransformMatrix[1][2] = T{-1} *	( a * f - c * d ) / m_det;
				m_InverseTransformMatrix[2][2] = 			( a * e - b * d ) / m_det;
			}

		public:
			ReferenceTransform(const tetrahedron_t& target_element) {
				m_d = target_element[3];
				for(auto i = size_type_t{0}; i < m_TransformMatrix.size(); ++i)
					m_TransformMatrix[i] = target_element[i];

				for(auto i = size_type_t{0}; i + 1 < target_element.size(); ++i)
					for(auto j = size_type_t{0}; j < size_type_t{3}; ++j)
						m_TransformMatrix[i][j] -= m_d[j];
				
				CalculateDeterminant();
				assert( std::abs( m_det ) > 5 * std::numeric_limits<T>::epsilon() );
				CalculateInverse();
			}

			auto GetDeterminantAbs() const {
				return std::abs( m_det );
			}

			auto operator()(const point_t& x0) const {
				auto xt = point_t{};
				for(auto i = size_type_t{0}; i < m_TransformMatrix.size(); ++i)
					for(auto j = size_type_t{0}; j < size_type_t{3}; ++j)
						xt[j] += m_TransformMatrix[i][j] * x0[j];
				
				for(auto j = size_type_t{0}; j < size_type_t{3}; ++j)
					xt[j] += m_d[j];

				return std::move(xt);
			}

			auto InverseMap(const point_t& p) const {
				// Do note that this is the transformed multiplication taking place.
				const auto trans_val = point_t{p[0] - m_d[0], p[1] - m_d[1], p[2] - m_d[2]};
				auto retval = point_t{};
				for(auto i = size_type_t{0}; i < size_type_t{3}; ++i) {
					retval[i] = 0;
					for(auto j = size_type_t{0}; j < size_type_t{3}; ++j)
						retval[i] += m_InverseTransformMatrix[j][i] * trans_val[j];
				}
				return std::move(retval);
			}

			auto TransformDerivative_Scalar(const point_t& grad_f) const {
				// Do note that this is the transformed multiplication taking place.
				auto retval = point_t{};
				for(auto i = size_type_t{0}; i < size_type_t{3}; ++i) {
					retval[i] = 0;
					for(auto j = size_type_t{0}; j < size_type_t{3}; ++j)
						retval[i] += m_InverseTransformMatrix[j][i] * grad_f[j];
				}
				return std::move(retval);
			}
		};
	}
}

#endif
