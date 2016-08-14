#include <type_traits>
#include <utility>
#include <array>

#include "Tetrahedron.hpp"
#include "QuadratureFormulas.hpp"

namespace QuadratureFormulas {

	namespace Tetrahedrons {

		template<typename T>
		struct Formula_3DT1 {
			const std::array<T, 1> weights{ T{1}/T{6} };
			const std::array<SpacePoint<T>, 1> points{ { T{1}/T{4}, T{1}/T{4}, T{1}/T{4} } };

			template<typename F>
			auto operator()(const F& f_integrand) {
				return EvaluateQuadrature(points, weights, f_integrand);
			}
		};

		template<typename T>
		struct Formula_3DT2 {
			const std::array<T, 4> weights{ T{1}/T{24}, T{1}/T{24}, T{1}/T{24}, T{1}/T{24} };
			const std::array<SpacePoint<T>, 4> points{ { T{0}, T{0}, T{0} }, { T{1}, T{0}, T{0} }, { T{0}, T{1}, T{0} }, { T{0}, T{0}, T{1} } };

			template<typename F>
			auto operator()(const F& f_integrand) {
				return EvaluateQuadrature(points, weights, f_integrand);
			}
		};

		template<typename T>
		struct Formula_3DT3 {
		private:
			constexpr static T sqrt_of_five = T{2.2360679774997896964091736687313};
		public:
			const std::array<T, 4> weights{ T{1}/T{24}, T{1}/T{24}, T{1}/T{24}, T{1}/T{24} };
			const std::array<SpacePoint<T>, 4> points{
				{ (T{5} - sqrt_of_five<T>)/T{20}, (T{5} - sqrt_of_five<T>)/T{20}, (T{5} - sqrt_of_five<T>)/T{20} }, 
				{ (T{5} + T{3} * sqrt_of_five<T>)/T{20}, (T{5} - sqrt_of_five<T>)/T{20}, (T{5} - sqrt_of_five<T>)/T{20} }, 
				{ (T{5} - sqrt_of_five<T>)/T{20}, (T{5} + T{3} * sqrt_of_five<T>)/T{20}, (T{5} - sqrt_of_five<T>)/T{20} }, 
				{ (T{5} - sqrt_of_five<T>)/T{20}, (T{5} - sqrt_of_five<T>)/T{20}, (T{5} + T{3} * sqrt_of_five<T>)/T{20} } 
			};

			template<typename F>
			auto operator()(const F& f_integrand) {
				return EvaluateQuadrature(points, weights, f_integrand);
			}
		};

		template<typename T>
		class ReferenceTransform {
		public:
			using point_t = Tetrahedron<T>::point_t;
		protected:
			std::array<point_t, 3> m_TransformMatrix;
			point_t m_d;
			T m_det;

			void CalculateDeterminant() {
				m_det = T{0};
				m_det += m_TransformMatrix[0].x * (m_TransformMatrix[1].y * m_TransformMatrix[2].z - m_TransformMatrix[2].y * m_TransformMatrix[1].z);
				m_det -= m_TransformMatrix[1].x * (m_TransformMatrix[0].y * m_TransformMatrix[2].z - m_TransformMatrix[2].y * m_TransformMatrix[0].z);
				m_det += m_TransformMatrix[2].x * (m_TransformMatrix[0].y * m_TransformMatrix[1].z - m_TransformMatrix[1].y * m_TransformMatrix[0].z); 
			}
		public:
			ReferenceTransform(const Tetrahedron<T>& target_element) {
				m_d = target_element.d;
				m_TransformMatrix[0] = target_element.a - m_d;
				m_TransformMatrix[1] = target_element.b - m_d;
				m_TransformMatrix[2] = target_element.c - m_d;
				CalculateDeterminant();
			}

			auto GetDeterminant() const {
				return m_det;
			}

			auto operator()(const point_t& x0) {
				auto xt = m_TransformMatrix[0] * x0.x;
				xt += m_TransformMatrix[1] * x0.y;
				xt += m_TransformMatrix[2] * x0.z;
				xt += m_d;
				return xt;
			}
		};
	}
}