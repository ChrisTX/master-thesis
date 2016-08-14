#include <type_traits>
#include <utility>
#include <array>

#include "QuadratureFormulas.hpp"

namespace QuadratureFormulas {

	namespace Triangles {

		template<typename T>
		struct Formula_2DD1 {
			const std::array<T, 1> weights{ T{1}/T{2} };
			const std::array<SurfacePoint<T>, 1> points{ { T{1}/T{3}, T{1}/T{3} } };

			template<typename F>
			auto operator()(const F& f_integrand) {
				return EvaluateQuadrature(points, weights, f_integrand);
			}
		};

		template<typename T>
		struct Formula_2DD2 {
			const std::array<T, 3> weights{ T{1}/T{6}, T{1}/T{6}, T{1}/T{6} };
			const std::array<SurfacePoint<T>, 3> points{ { T{0}, T{0} }, { T{1}, T{0} }, { T{0}, T{1} } };

			template<typename F>
			auto operator()(const F& f_integrand) {
				return EvaluateQuadrature(points, weights, f_integrand);
			}
		};

		template<typename T>
		struct Formula_2DD3 {
			const std::array<T, 3> weights{ T{1}/T{6}, T{1}/T{6}, T{1}/T{6} };
			const std::array<SurfacePoint<T>, 3> points{ { T{1}/T{2}, T{0} }, { T{1}/T{2}, T{1}/T{2} }, { T{0}, T{1}/T{2} } };

			template<typename F>
			auto operator()(const F& f_integrand) {
				return EvaluateQuadrature(points, weights, f_integrand);
			}
		};

		template<typename T>
		struct Formula_2DD4 {
			const std::array<T, 3> weights{ T{1}/T{6}, T{1}/T{6}, T{1}/T{6} };
			const std::array<SurfacePoint<T>, 3> points{ { T{1}/T{6}, T{1}/T{6} }, { T{4}/T{6}, T{1}/T{6} }, { T{1}/T{6}, T{4}/T{6} } };

			template<typename F>
			auto operator()(const F& f_integrand) {
				return EvaluateQuadrature(points, weights, f_integrand);
			}
		};


		template<typename T>
		struct Formula_2DD5 {
			const std::array<T, 7> weights{ T{3}/T{120}, T{3}/T{120}, T{3}/T{120}, T{8}/T{120}, T{8}/T{120}, T{8}/T{120}, T{27}/T{120} };
			const std::array<SurfacePoint<T>, 7> points{ { T{0}, T{0} }, { T{1}, T{0} }, { T{0}, T{1} }, { T{1}/T{2}, T{0} }, { T{1}/T{2}, T{1}/T{2} }, { T{0}, T{1}/T{2} }, { T{1}/T{3}, T{1}/T{3} } };

			template<typename F>
			auto operator()(const F& f_integrand) {
				return EvaluateQuadrature(points, weights, f_integrand);
			}
		};


		template<typename T>
		struct Formula_2DD6 {
		private:
			constexpr static T sqrt_of_fifteen = T{3.8729833462074168851792653997824};
			constexpr static T vma = (T{6} - sqrt_of_fifteen<T>)/T{21};
			constexpr static T vpa = (T{6} + sqrt_of_fifteen<T>)/T{21};
			constexpr static T vmb = (T{9} + T{2} * sqrt_of_fifteen<T>)/T{21};
			constexpr static T vpb = (T{9} - T{2} * sqrt_of_fifteen<T>)/T{21};
			constexpr static T wa = (T{155} - sqrt_of_fifteen<T>)/T{2400};
			constexpr static T wb = (T{155} + sqrt_of_fifteen<T>)/T{2400};
		public:
			const std::array<T, 7> weights{ wa, wa, wa, wb, wb, wb, T{9}/T{80} };
			const std::array<SurfacePoint<T>, 7> points{
				{ vma, vma }, { vpb, vma }, { vma, vpb }, { vpa, vmb }, { vpa, vpa }, { vmb, vpa }, { T{1}/T{3}, T{1}/T{3} }
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
			using surf_point_t = Tetrahedron<T>::surf_point_t;
		protected:
			std::array<point_t, 2> m_TransformMatrix;
			point_t m_c;
			T m_det;

			void CalculateDeterminant() {
				const auto det11 = m_TransformMatrix[0].x * m_TransformMatrix[0].x + m_TransformMatrix[0].y * m_TransformMatrix[0].y + m_TransformMatrix[0].z * m_TransformMatrix[0].z;
				const auto det12 = m_TransformMatrix[0].x * m_TransformMatrix[1].x + m_TransformMatrix[0].y * m_TransformMatrix[1].y + m_TransformMatrix[0].z * m_TransformMatrix[1].z;
				const auto det21 = det12;
				const auto det22 = m_TransformMatrix[1].x * m_TransformMatrix[1].x + m_TransformMatrix[1].y * m_TransformMatrix[1].y + m_TransformMatrix[1].z * m_TransformMatrix[1].z;
				m_det = det11 * det22 - det21 * det12;
			}
		public:
			ReferenceTransform(const SpaceTriangle<T>& target_element) {
				m_c = target_element.c;
				m_TransformMatrix[0] = target_element.a - m_c;
				m_TransformMatrix[1] = target_element.b - m_c;
				CalculateDeterminant();
			}

			auto GetDeterminant() const {
				return m_det;
			}

			auto operator()(const surf_point_t& x0) {
				auto xt = m_TransformMatrix[0] * x0.x;
				xt += m_TransformMatrix[1] * x0.y;
				xt += m_c;
				return xt;
			}
		};
	}
}