#ifndef SPACE_TIME_OPTIMIZER_HPP
#define SPACE_TIME_OPTIMIZER_HPP

#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "TetrahedralMesh.hpp"
#include "CSRMatrix.hpp"

#ifdef HAVE_MKL
#include "IterativeSolvers.hpp"
#endif

#include "TriangularQuadrature.hpp"
#include "TetrahedralQuadrature.hpp"

#include <vtkTetra.h>
#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPointData.h>
#include <vtkXMLUnstructuredGridWriter.h>

template<class T, class TriangQuadFm, class TetraQuadFm>
class STMFormEvaluator {
protected:
	const TetrahedralMesh<T>& m_Mesh;
	const T sigma;
	const T alpha;
	const T beta;
	const T lambda;
	const T theta;

	template<typename TK>
	static auto x_scal_eval(const TK& u, const TK& v) {
		return u[0] * v[0] + u[1] * v[1];
	}

	const TriangQuadFm triang_quadfm{};
	const TetraQuadFm tetra_quadfm{};

public:
	using ElementId_t = typename TetrahedralMesh<T>::ElementId_t;
	using SurfaceId_t = typename TetrahedralMesh<T>::SurfaceId_t;
	using Point_t = typename TetrahedralMesh<T>::Point_t;

	STMFormEvaluator(const TetrahedralMesh<T>& mesh, const T par_sigma, const T par_alpha, const T par_beta, const T par_lambda, const T par_theta) :
		m_Mesh{ mesh }, sigma{ par_sigma }, alpha{ par_alpha }, beta{ par_beta }, lambda{ par_lambda }, theta{ par_theta }
	{
		if (!std::isfinite(lambda) || !std::isfinite(beta) || lambda < std::numeric_limits<T>::epsilon())
			throw std::invalid_argument("Invalid parameters");
	}

	template<class BasisFuncs, class BasisIndex>
	auto EvaluateAh_Element(const ElementId_t elemid, const BasisFuncs bf, const BasisIndex biv, const BasisIndex biu) const {
		const auto cur_tetrahedron = m_Mesh.ElementIdToTetrahedron(elemid);
		const auto ref_tran = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(cur_tetrahedron);
		const auto ref_tran_det = ref_tran.GetDeterminantAbs();

		const auto integrand = [&](const auto& sp) -> auto {
			const auto p_space = ref_tran(sp);

			const auto du = ref_tran.EvaluateTransformedBasisDerivative(bf, biu, p_space);
			const auto dv = ref_tran.EvaluateTransformedBasisDerivative(bf, biv, p_space);

			return x_scal_eval(du, dv); 
		};

		auto intval = tetra_quadfm(integrand) * ref_tran_det;

		// At this point, first_intval is the part of the bilinear form that isspace integral.
		// We need to add the three edge terms now
		const auto inner_integrand_part = [&](const auto& surfid, const auto& surfdata) -> auto {
			const auto surf_nm = surfdata.normal_vector_by_elemid(elemid);

			const auto cur_triangle = m_Mesh.SurfaceIdToTriangle(surfid);
			const auto ref_triang_tran = QuadratureFormulas::Triangles::ReferenceTransform<T>(cur_triangle);

			const auto integrand_fn = [&](const auto& sp) -> auto {
				const auto p_space = ref_triang_tran(sp);

				const auto uh = ref_tran.EvaluateTransformedBasis(bf, biu, p_space);
				const auto vh = ref_tran.EvaluateTransformedBasis(bf, biv, p_space);
				const auto du = ref_tran.EvaluateTransformedBasisDerivative(bf, biu, p_space);
				const auto dv = ref_tran.EvaluateTransformedBasisDerivative(bf, biv, p_space);

				const auto dux_n = x_scal_eval(du, surf_nm);
				const auto dvx_n = x_scal_eval(dv, surf_nm);

				// First two interface integral terms
				const auto f1 = T{-1} * ( (T{1}/T{2}) * (dux_n * vh + uh * dvx_n) );
				// Third penalty term
				const auto f2 = ( sigma/surfdata.h ) * ( uh * vh * x_scal_eval(surf_nm, surf_nm) );
				return f1 + f2;
			};

			const auto ref_triang_tran_det = ref_triang_tran.GetDeterminantSqrt();
			return triang_quadfm(integrand_fn) * ref_triang_tran_det;
		};

		const auto surface_integral_part = [&](const auto& surfid) -> auto {
			const auto cur_triangle = m_Mesh.SurfaceIdToTriangle(surfid);
			const auto ref_triang_tran = QuadratureFormulas::Triangles::ReferenceTransform<T>(cur_triangle);

			const auto integrand_fn = [&](auto sp) -> auto {
				const auto p_space = ref_triang_tran(sp);

				const auto uh = ref_tran.EvaluateTransformedBasis(bf, biu, p_space);
				const auto vh = ref_tran.EvaluateTransformedBasis(bf, biv, p_space);

				return uh * vh;
			};

			const auto ref_triang_tran_det = ref_triang_tran.GetDeterminantSqrt();
			return triang_quadfm(integrand_fn) * ref_triang_tran_det;
		};

		const auto& curelem = m_Mesh.m_ElementList[elemid];
		const auto a = curelem.corners[0];
		const auto b = curelem.corners[1];
		const auto c = curelem.corners[2];
		const auto d = curelem.corners[3];

		for(const auto si : { SurfaceId_t{a, b, c}, SurfaceId_t{a, b, d}, SurfaceId_t{b, c, d}, SurfaceId_t{a, c, d} }) {
			const auto& surfdata = m_Mesh.SurfaceDataById(si);
			if (surfdata.type == TetrahedralMesh<T>::SurfaceType_t::Inner)
				intval += inner_integrand_part(si, surfdata);
			else if (surfdata.type == TetrahedralMesh<T>::SurfaceType_t::MidTime)
				intval += alpha * surface_integral_part(si);
		}
		return theta * intval;
	}

	template<class BasisFuncs, class BasisIndex>
	auto EvaluateAh_Surface(const SurfaceId_t& surfid, const BasisFuncs& bf, const BasisIndex biv, const BasisIndex biu) const {
		// We assume that uh is an element of the second tetrahedron of the surface
		// and that vh is an element of the first tetrahedron, respectively.
		const auto& cur_surface_data = m_Mesh.SurfaceDataById(surfid);
		assert( cur_surface_data.type == TetrahedralMesh<T>::SurfaceType_t::Inner );

		const auto& elemid_v = cur_surface_data.adjacent_elements[0];
		const auto& tetrahedr_v = m_Mesh.ElementIdToTetrahedron(elemid_v);
		const auto& ref_tran_v = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(tetrahedr_v);
		const auto surf_nm_v = cur_surface_data.normal_vector_by_elemid(elemid_v);

		const auto& elemid_u = cur_surface_data.adjacent_elements[1];
		const auto& tetrahedr_u = m_Mesh.ElementIdToTetrahedron(elemid_u);
		const auto& ref_tran_u = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(tetrahedr_u);
		const auto surf_nm_u = cur_surface_data.normal_vector_by_elemid(elemid_u);

		const auto cur_triangle = m_Mesh.SurfaceIdToTriangle(surfid);
		const auto ref_triang_tran = QuadratureFormulas::Triangles::ReferenceTransform<T>(cur_triangle);
		const auto ref_triang_tran_det = ref_triang_tran.GetDeterminantSqrt();

		const auto integrand_fn = [&](const auto& sp) -> auto {
			const auto p_space = ref_triang_tran(sp);

			const auto uh = ref_tran_u.EvaluateTransformedBasis(bf, biu, p_space);
			const auto vh = ref_tran_v.EvaluateTransformedBasis(bf, biv, p_space);
			const auto du = ref_tran_u.EvaluateTransformedBasisDerivative(bf, biu, p_space);
			const auto dv = ref_tran_v.EvaluateTransformedBasisDerivative(bf, biv, p_space);

			const auto dux_nv = x_scal_eval(du, surf_nm_v);
			const auto dvx_nu = x_scal_eval(dv, surf_nm_u);

			// First two interface integral terms
			const auto f1 = T{-1} * ( (T{1}/T{2}) * ( dux_nv * vh + uh * dvx_nu ) );
			// Third penalty term
			const auto f2 = ( sigma/cur_surface_data.h ) * ( uh * vh * x_scal_eval(surf_nm_u, surf_nm_v) );
			return f1 + f2;
		};

		return theta * triang_quadfm(integrand_fn) * ref_triang_tran_det;
	}

	template<class BasisFuncs, class BasisIndex>
	auto EvaluateBh_Element(const ElementId_t elemid, const BasisFuncs& bf, const BasisIndex biv, const BasisIndex biu) const {
		const auto cur_tetrahedron = m_Mesh.ElementIdToTetrahedron(elemid);
		const auto ref_tran = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(cur_tetrahedron);
		const auto ref_tran_det = ref_tran.GetDeterminantAbs();

		const auto integrand = [&](const auto& sp) -> auto {
			const auto p_space = ref_tran(sp);

			const auto uh = ref_tran.EvaluateTransformedBasis(bf, biu, p_space);
			const auto dv = ref_tran.EvaluateTransformedBasisDerivative(bf, biv, p_space);

			return uh * dv[2]; 
		};

		auto intval = T{-1} * tetra_quadfm(integrand) * ref_tran_det;

		// At this point, first_intval is the part of the bilinear form that isspace integral.
		// We need to add the three edge terms now
		const auto surface_integral_part = [&](const auto& surfid) -> auto {
			const auto cur_triangle = m_Mesh.SurfaceIdToTriangle(surfid);
			const auto ref_triang_tran = QuadratureFormulas::Triangles::ReferenceTransform<T>(cur_triangle);

			const auto integrand_fn = [&](auto sp) -> auto {
				const auto p_space = ref_triang_tran(sp);

				const auto uh = ref_tran.EvaluateTransformedBasis(bf, biu, p_space);
				const auto vh = ref_tran.EvaluateTransformedBasis(bf, biv, p_space);

				return uh * vh;
			};

			const auto ref_triang_tran_det = ref_triang_tran.GetDeterminantSqrt();
			return triang_quadfm(integrand_fn) * ref_triang_tran_det;
		};

		const auto inner_integral_part = [&](const auto& surfid, const auto& surfdata) -> auto {
			// Given our piecewise polynomial approach, {uh}^up = uh if the time normal is > 0 and 0 otherwise
			if( surfdata.is_time_orthogonal || surfdata.get_upstream_element() != elemid )
				return T{0};

			const auto surf_nm = surfdata.normal_vector_by_elemid(elemid);

			const auto cur_triangle = m_Mesh.SurfaceIdToTriangle(surfid);
			const auto ref_triang_tran = QuadratureFormulas::Triangles::ReferenceTransform<T>(cur_triangle);

			const auto integrand_fn = [&](auto sp) -> auto {
				const auto p_space = ref_triang_tran(sp);

				const auto uh = ref_tran.EvaluateTransformedBasis(bf, biu, p_space);
				const auto vh = ref_tran.EvaluateTransformedBasis(bf, biv, p_space);

				return uh * vh * surf_nm[2];
			};

			const auto ref_triang_tran_det = ref_triang_tran.GetDeterminantSqrt();
			return triang_quadfm(integrand_fn) * ref_triang_tran_det;
		};

		const auto& curelem = m_Mesh.m_ElementList[elemid];
		const auto a = curelem.corners[0];
		const auto b = curelem.corners[1];
		const auto c = curelem.corners[2];
		const auto d = curelem.corners[3];

		for(const auto si : { SurfaceId_t{a, b, c}, SurfaceId_t{a, b, d}, SurfaceId_t{b, c, d}, SurfaceId_t{a, c, d} }) {
			const auto& surfdata = m_Mesh.SurfaceDataById(si);
			if (surfdata.type == TetrahedralMesh<T>::SurfaceType_t::EndTime)
				intval += surface_integral_part( si );
			else if( surfdata.type == TetrahedralMesh<T>::SurfaceType_t::Inner )
				intval += inner_integral_part( si, surfdata );
			else if( surfdata.type == TetrahedralMesh<T>::SurfaceType_t::Undefined )
				assert(false);
		}
		return intval;
	}

	template<class BasisFuncs, class BasisIndex>
	auto EvaluateBh_Surface(const SurfaceId_t& surfid, const BasisFuncs& bf, const BasisIndex biv, const BasisIndex biu) const {
		// Note that this function can be used to sum both bh and bh' by interpreting its returns differently

		// We assume that uh is an element of the first tetrahedron of the surface
		// and that vh is an element of the second tetrahedron, respectively.
		const auto& cur_surface_data = m_Mesh.SurfaceDataById(surfid);
		assert( cur_surface_data.type == TetrahedralMesh<T>::SurfaceType_t::Inner );

		const auto elemid_down = cur_surface_data.get_downstream_element();
		// If the element is time orthogonal, {{uh}}^up is zero by definition and we can stop.
		// Otherwise, the form is not symmetrical. {{uh}}^up = uh for the upper element and zero for the lower one.
		// Hence this form is NOT SYMMETRICAL. It needs to be added to the appropriate matrix position.
		if( cur_surface_data.is_time_orthogonal )
			return T{0};

		const auto elemid_up = cur_surface_data.get_upstream_element();
		const auto tetrahedr_down = m_Mesh.ElementIdToTetrahedron(elemid_down);
		const auto ref_tran_down = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(tetrahedr_down);
		const auto tetrahedr_up = m_Mesh.ElementIdToTetrahedron(elemid_up);
		const auto ref_tran_up = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(tetrahedr_up);

		const auto surf_nm_down = cur_surface_data.normal_vector_by_elemid(elemid_down);

		const auto cur_triangle = m_Mesh.SurfaceIdToTriangle(surfid);
		const auto ref_triang_tran = QuadratureFormulas::Triangles::ReferenceTransform<T>(cur_triangle);
		const auto ref_triang_tran_det = ref_triang_tran.GetDeterminantSqrt();

		const auto integrand_fn = [&](const auto& sp) -> auto {
			const auto p_space = ref_triang_tran(sp);

			const auto uh = ref_tran_up.EvaluateTransformedBasis(bf, biu, p_space);
			const auto vh = ref_tran_down.EvaluateTransformedBasis(bf, biv, p_space);
			
			return uh * vh * surf_nm_down[2];
		};

		return triang_quadfm(integrand_fn) * ref_triang_tran_det;
	}

	template<class BasisFuncs, class BasisIndex>
	auto EvaluateJh_Surface(const SurfaceId_t& surfid, const BasisFuncs& bf, const BasisIndex biv, const BasisIndex biu) const {
		const auto& cur_surface_data = m_Mesh.SurfaceDataById(surfid);
		assert(cur_surface_data.type == TetrahedralMesh<T>::SurfaceType_t::MidTime);

		const auto adj_elem_id = cur_surface_data.adjacent_elements[0];
		const auto cur_tetrahedron = m_Mesh.ElementIdToTetrahedron(adj_elem_id);
		const auto ref_tran = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(cur_tetrahedron);

		const auto surface_integral_part = [&]() -> auto {
			const auto cur_triangle = m_Mesh.SurfaceIdToTriangle(surfid);
			const auto ref_triang_tran = QuadratureFormulas::Triangles::ReferenceTransform<T>(cur_triangle);

			const auto integrand_fn = [&](auto sp) -> auto {
				const auto p_space = ref_triang_tran(sp);

				const auto uh = ref_tran.EvaluateTransformedBasis(bf, biu, p_space);
				const auto vh = ref_tran.EvaluateTransformedBasis(bf, biv, p_space);

				return uh * vh;
			};

			const auto ref_triang_tran_det = ref_triang_tran.GetDeterminantSqrt();
			return triang_quadfm(integrand_fn) * ref_triang_tran_det;
		};

		return beta * beta * surface_integral_part() / lambda;
	}

	template<class BasisFuncs, class BasisIndex>
	auto EvaluateKh_Surface(const SurfaceId_t& surfid, const BasisFuncs& bf, const BasisIndex biv, const BasisIndex biu) const {
		const auto& cur_surface_data = m_Mesh.SurfaceDataById(surfid);
		assert(cur_surface_data.type == TetrahedralMesh<T>::SurfaceType_t::EndTime);

		const auto adj_elem_id = cur_surface_data.adjacent_elements[0];
		const auto cur_tetrahedron = m_Mesh.ElementIdToTetrahedron(adj_elem_id);
		const auto ref_tran = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(cur_tetrahedron);

		const auto surface_integral_part = [&]() -> auto {
			const auto cur_triangle = m_Mesh.SurfaceIdToTriangle(surfid);
			const auto ref_triang_tran = QuadratureFormulas::Triangles::ReferenceTransform<T>(cur_triangle);

			const auto integrand_fn = [&](auto sp) -> auto {
				const auto p_space = ref_triang_tran(sp);

				const auto uh = ref_tran.EvaluateTransformedBasis(bf, biu, p_space);
				const auto vh = ref_tran.EvaluateTransformedBasis(bf, biv, p_space);

				return uh * vh;
			};

			const auto ref_triang_tran_det = ref_triang_tran.GetDeterminantSqrt();
			return triang_quadfm(integrand_fn) * ref_triang_tran_det;
		};

		return T{ -1 } *surface_integral_part();
	}

	template<class BasisFuncs, class BasisIndex>
	auto EvaluateKh_Inner_Surface(const SurfaceId_t& surfid, const BasisFuncs& bf, const BasisIndex biv, const BasisIndex biu) const {
		const auto& cur_surface_data = m_Mesh.SurfaceDataById(surfid);
		assert(cur_surface_data.type == TetrahedralMesh<T>::SurfaceType_t::MidTime);

		const auto adj_elem_id = cur_surface_data.adjacent_elements[0];
		const auto cur_tetrahedron = m_Mesh.ElementIdToTetrahedron(adj_elem_id);
		const auto ref_tran = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(cur_tetrahedron);

		const auto surface_integral_part = [&]() -> auto {
			const auto cur_triangle = m_Mesh.SurfaceIdToTriangle(surfid);
			const auto ref_triang_tran = QuadratureFormulas::Triangles::ReferenceTransform<T>(cur_triangle);

			const auto integrand_fn = [&](auto sp) -> auto {
				const auto p_space = ref_triang_tran(sp);

				const auto uh = ref_tran.EvaluateTransformedBasis(bf, biu, p_space);
				const auto vh = ref_tran.EvaluateTransformedBasis(bf, biv, p_space);

				return uh * vh;
			};

			const auto ref_triang_tran_det = ref_triang_tran.GetDeterminantSqrt();
			return triang_quadfm(integrand_fn) * ref_triang_tran_det;
		};

		return T{ -1 } * surface_integral_part();
	}

	template<class BasisFuncs, class BasisIndex>
	auto EvaluateJh_Inner_Element(const ElementId_t elemid, const BasisFuncs bf, const BasisIndex biv, const BasisIndex biu) const {
		const auto cur_tetrahedron = m_Mesh.ElementIdToTetrahedron(elemid);
		const auto ref_tran = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(cur_tetrahedron);
		const auto ref_tran_det = ref_tran.GetDeterminantAbs();

		const auto integrand = [&](const auto& sp) -> auto {
			const auto p_space = ref_tran(sp);

			const auto uh = ref_tran.EvaluateTransformedBasis(bf, biu, p_space);
			const auto vh = ref_tran.EvaluateTransformedBasis(bf, biv, p_space);

			return uh * vh;
		};

		return beta * beta * tetra_quadfm(integrand) * ref_tran_det / lambda;
	}

	template<class BasisFuncs, class BasisIndex>
	auto EvaluateKh_Symmetric_Element(const ElementId_t elemid, const BasisFuncs bf, const BasisIndex biv, const BasisIndex biu) const {
		const auto cur_tetrahedron = m_Mesh.ElementIdToTetrahedron(elemid);
		const auto ref_tran = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(cur_tetrahedron);
		const auto ref_tran_det = ref_tran.GetDeterminantAbs();

		const auto integrand = [&](const auto& sp) -> auto {
			const auto p_space = ref_tran(sp);

			const auto uh = ref_tran.EvaluateTransformedBasis(bf, biu, p_space);
			const auto vh = ref_tran.EvaluateTransformedBasis(bf, biv, p_space);

			return uh * vh;
		};

		return T{ -1 } * tetra_quadfm(integrand) * ref_tran_det;
	}

	template<class F, class BasisFuncs, class BasisIndex>
	auto EvaluateLV_Surface(const F& rhs_func, const SurfaceId_t& surfid, const BasisFuncs& bf, const BasisIndex biv) const {
		const auto& cur_surface_data = m_Mesh.SurfaceDataById(surfid);
		assert(cur_surface_data.type == TetrahedralMesh<T>::SurfaceType_t::StartTime 
			|| cur_surface_data.type == TetrahedralMesh<T>::SurfaceType_t::EndTime
			|| cur_surface_data.type == TetrahedralMesh<T>::SurfaceType_t::MidTime);

		const auto adj_elem_id = cur_surface_data.adjacent_elements[0];
		const auto cur_tetrahedron = m_Mesh.ElementIdToTetrahedron(adj_elem_id);
		const auto ref_tran = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(cur_tetrahedron);

		const auto surface_integral_part = [&]() -> auto {
			const auto cur_triangle = m_Mesh.SurfaceIdToTriangle(surfid);
			const auto ref_triang_tran = QuadratureFormulas::Triangles::ReferenceTransform<T>(cur_triangle);

			const auto integrand_fn = [&](auto sp) -> auto {
				const auto p_space = ref_triang_tran(sp);

				const auto uh = rhs_func(p_space[0], p_space[1], p_space[2]);
				const auto vh = ref_tran.EvaluateTransformedBasis(bf, biv, p_space);

				return uh * vh;
			};

			const auto ref_triang_tran_det = ref_triang_tran.GetDeterminantSqrt();
			return triang_quadfm(integrand_fn) * ref_triang_tran_det;
		};

		return surface_integral_part();
	}
};

template<class T, class TriangQuadFm, class TetraQuadFm>
struct STMAssembler : public STMFormEvaluator<T, TriangQuadFm, TetraQuadFm> {
	using ElementId_t = typename STMFormEvaluator<T, TriangQuadFm, TetraQuadFm>::ElementId_t;
	using SurfaceId_t = typename STMFormEvaluator<T, TriangQuadFm, TetraQuadFm>::SurfaceId_t;
	using SurfaceType_t = typename TetrahedralMesh<T>::SurfaceType_t;
	using Point_t = typename TetrahedralMesh<T>::Point_t;

	STMAssembler(const TetrahedralMesh<T>& mesh, const T par_sigma, const T par_alpha, const T par_beta, const T par_lambda, const T par_theta) :
		STMFormEvaluator<T, TriangQuadFm, TetraQuadFm>( mesh, par_sigma, par_alpha, par_beta, par_lambda, par_theta )
	{
	}

	template<class BasisFuncs, class F0, class FT>
	auto AssembleLV_Boundary(const F0& y0, const FT& yOmega) const {
		const auto basis_f = BasisFuncs{};
		using basis_index_t = typename BasisFuncs::index_t;
		using basis_und_t = std::underlying_type_t<basis_index_t>;
		const auto num_elems = this->m_Mesh.m_ElementList.size();
		const auto num_basis = basis_f.size();
		const auto block_size = num_basis * num_elems;
		const auto matrix_dim = 2 * block_size;

		auto loadvec = std::vector<T>(matrix_dim);

		for (const auto& pval : this->m_Mesh.m_SurfaceList) {
			const auto& surf_id = pval.first;
			const auto& surf_data = pval.second;
			switch (surf_data.type) {
				case SurfaceType_t::EndTime:
					{
						const auto start_offset = surf_data.adjacent_elements[0] * num_basis;
						for (auto bi = basis_und_t{ 0 }; bi < num_basis; ++bi) {
							const auto offset_vi = start_offset + bi;

							const auto form_val_LV_low = this->EvaluateLV_Surface(yOmega, surf_id, basis_f, static_cast<basis_index_t>(bi));
							assert(std::isfinite(form_val_LV_low));
							loadvec[offset_vi] -= form_val_LV_low;
						}
					}
					break;

				case SurfaceType_t::StartTime:
					{
						const auto start_offset = surf_data.adjacent_elements[0] * num_basis;
						for (auto bi = basis_und_t{ 0 }; bi < num_basis; ++bi) {
							const auto offset_vi = start_offset + bi;

							const auto form_val_LV_up = this->EvaluateLV_Surface(y0, surf_id, basis_f, static_cast<basis_index_t>(bi));
							assert(std::isfinite(form_val_LV_up));
							loadvec[block_size + offset_vi] += form_val_LV_up;
						}
					}
					break;

				case SurfaceType_t::Inner:
					break;

				case SurfaceType_t::MidTime:
					break;

				case SurfaceType_t::Undefined:
					assert(false);
			}
		}

		return std::move(loadvec);
	}

	template<class BasisFuncs, class FT>
	auto AssembleLV_Inner(const FT& ySigma) const {
		const auto basis_f = BasisFuncs{};
		using basis_index_t = typename BasisFuncs::index_t;
		using basis_und_t = std::underlying_type_t<basis_index_t>;
		const auto num_elems = this->m_Mesh.m_ElementList.size();
		const auto num_basis = basis_f.size();
		const auto block_size = num_basis * num_elems;
		const auto matrix_dim = 2 * block_size;

		auto loadvec = std::vector<T>(matrix_dim);

		for (const auto& pval : this->m_Mesh.m_SurfaceList) {
			const auto& surf_id = pval.first;
			const auto& surf_data = pval.second;
			switch (surf_data.type) {
			case SurfaceType_t::EndTime:
				break;

			case SurfaceType_t::StartTime:
				break;

			case SurfaceType_t::Inner:
				break;

			case SurfaceType_t::MidTime:
			{
				const auto start_offset = surf_data.adjacent_elements[0] * num_basis;
				for (auto bi = basis_und_t{ 0 }; bi < num_basis; ++bi) {
					const auto offset_vi = start_offset + bi;

					const auto form_val_LV_up = this->EvaluateLV_Surface(ySigma, surf_id, basis_f, static_cast<basis_index_t>(bi));
					assert(std::isfinite(form_val_LV_up));
					loadvec[offset_vi] -= form_val_LV_up;
				}
			}
			break;

			case SurfaceType_t::Undefined:
				assert(false);
			}
		}

		return std::move(loadvec);
	}

	template<class BasisFuncs, class FT, class FT2, class FT3>
	auto AssembleLV_Symmetric(const FT& innerF, const FT2& yQ, const FT3& y0) const {
		const auto basis_f = BasisFuncs{};
		using basis_index_t = typename BasisFuncs::index_t;
		using basis_und_t = std::underlying_type_t<basis_index_t>;
		const auto num_elems = this->m_Mesh.m_ElementList.size();
		const auto num_basis = basis_f.size();
		const auto block_size = num_basis * num_elems;
		const auto matrix_dim = 2 * block_size;

		auto loadvec = std::vector<T>(matrix_dim);

		const auto inner_element_f = [&](const auto& rhs_func, const auto elemid, const auto biv) -> auto {
			const auto cur_tetrahedron = m_Mesh.ElementIdToTetrahedron(elemid);
			const auto ref_tran = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(cur_tetrahedron);
			const auto ref_tran_det = ref_tran.GetDeterminantAbs();

			const auto integrand = [&](const auto& sp) -> auto {
				const auto p_space = ref_tran(sp);

				const auto uh = rhs_func(p_space[0], p_space[1], p_space[2]);
				const auto vh = ref_tran.EvaluateTransformedBasis(basis_f, biv, p_space);

				return uh * vh;
			};

			return tetra_quadfm(integrand) * ref_tran_det;
		};

		for (auto i = ElementId_t{ 0 }; i < num_elems; ++i) {
			const auto start_offset = i * num_basis;
			for (auto bi = basis_und_t{ 0 }; bi < num_basis; ++bi) {
				const auto offset_vi = start_offset + bi;

				const auto form_val_rhs = inner_element_f(innerF, i, static_cast<basis_index_t>(bi));
				assert(std::isfinite(form_val_rhs));
				loadvec[block_size + offset_vi] += form_val_rhs;

				const auto form_val_rhsQ = inner_element_f(yQ, i, static_cast<basis_index_t>(bi));
				assert(std::isfinite(form_val_rhsQ));
				loadvec[offset_vi] -= form_val_rhsQ;
			}
		}

		for (const auto& pval : this->m_Mesh.m_SurfaceList) {
			const auto& surf_id = pval.first;
			const auto& surf_data = pval.second;
			switch (surf_data.type) {
			case SurfaceType_t::EndTime:
				break;

			case SurfaceType_t::StartTime:
			{
				const auto start_offset = surf_data.adjacent_elements[0] * num_basis;
				for (auto bi = basis_und_t{ 0 }; bi < num_basis; ++bi) {
					const auto offset_vi = start_offset + bi;

					const auto form_val_LV_up = this->EvaluateLV_Surface(y0, surf_id, basis_f, static_cast<basis_index_t>(bi));
					assert(std::isfinite(form_val_LV_up));
					loadvec[block_size + offset_vi] += form_val_LV_up;
				}
			}
				break;

			case SurfaceType_t::Inner:
				break;

			case SurfaceType_t::MidTime:
				break;

			case SurfaceType_t::Undefined:
				assert(false);
			}
		}

		// We need to ensure homogenous Dirichlet boundary conditions
		for (const auto& pval : this->m_Mesh.m_SurfaceList) {
			const auto& surf_id = pval.first;
			const auto& surf_data = pval.second;
			switch (surf_data.type) {
			case SurfaceType_t::Inner:
				break;

			case SurfaceType_t::MidTime:
			{
				const auto start_offset = surf_data.adjacent_elements[0] * num_basis;
				for (auto bi = basis_und_t{ 0 }; bi < num_basis; ++bi) {
					const auto offset_vi = start_offset + bi;

					const auto form_val_L2surf = this->EvaluateJh_Surface(surf_id, basis_f, static_cast<basis_index_t>(bi), static_cast<basis_index_t>(bi));
					assert(std::isfinite(form_val_L2surf));
					if (form_val_L2surf > 5 * std::numeric_limits<T>::epsilon() ) {
						loadvec[offset_vi] = T{ 0 };
						loadvec[block_size + offset_vi] = T{ 0 };
					}
				}
			}
			break;

			case SurfaceType_t::EndTime:
				break;

			case SurfaceType_t::StartTime:
				break;

			case SurfaceType_t::Undefined:
				assert(false);
			}
		}

		return std::move(loadvec);
	}

	template<class BasisFuncs>
	auto AssembleMatrix_Base() const {
		// In a dG approach, we have a given amount of functions ( BasisFuncs' size ) per element
		// Hence, the number of *active* elements in the mesh times the BasisFuncs is what we're looking for.

		const auto basis_f = BasisFuncs{};
		using basis_index_t = typename BasisFuncs::index_t;
		using basis_und_t = std::underlying_type_t<basis_index_t>;
		const auto num_elems = this->m_Mesh.m_ElementList.size();
		const auto num_basis = basis_f.size();
		using csr_size_t = typename Utility::CSRMatrixAssembler<T>::size_type;
		const auto block_size = static_cast<csr_size_t>(num_basis * num_elems);
		const auto matrix_dim = static_cast<csr_size_t>(2 * block_size);

		auto matassembler = Utility::CSRMatrixAssembler<T>{ matrix_dim, matrix_dim };

#ifdef SYMMETRIC_ASSEMBLY
		for (auto i = csr_size_t{ 0 }; i < matrix_dim; ++i)
			matassembler(i, i) = T{ 0 };
#endif

		// We first sum Ah + Bh on the inner-element interfaces up
		for (auto i = ElementId_t{ 0 }; i < num_elems; ++i) {
			const auto start_offset = i * num_basis;
			for (auto bi = basis_und_t{ 0 }; bi < num_basis; ++bi) {
				for (auto bj = bi; bj < num_basis; ++bj) {
					const auto offset_vi = static_cast<csr_size_t>(start_offset + bi);
					const auto offset_uj = static_cast<csr_size_t>(start_offset + bj);

					const auto form_val_Ah = this->EvaluateAh_Element(i, basis_f, static_cast<basis_index_t>(bi), static_cast<basis_index_t>(bj));
					assert(std::isfinite(form_val_Ah));

#ifndef SYMMETRIC_ASSEMBLY
					matassembler(block_size + offset_vi, offset_uj) = form_val_Ah;
#endif
					matassembler(offset_uj, block_size + offset_vi) = form_val_Ah;

					if (bi != bj) {
#ifndef SYMMETRIC_ASSEMBLY
						matassembler(block_size + offset_uj, offset_vi) = form_val_Ah;
#endif
						matassembler(offset_vi, block_size + offset_uj) = form_val_Ah;
					}
				}

				for (auto bj = basis_und_t{ 0 }; bj < num_basis; ++bj) {
					const auto offset_vi = static_cast<csr_size_t>(start_offset + bi);
					const auto offset_uj = static_cast<csr_size_t>(start_offset + bj);

					const auto form_val_Bh = this->EvaluateBh_Element(i, basis_f, static_cast<basis_index_t>(bi), static_cast<basis_index_t>(bj));
					assert(std::isfinite(form_val_Bh));
#ifndef SYMMETRIC_ASSEMBLY
					matassembler(block_size + offset_vi, offset_uj) += form_val_Bh;
#endif
					matassembler(offset_uj, block_size + offset_vi) += form_val_Bh;
				}
			}
		}

		// Aside from these per element integrals, we have some interface ones:
		// Ah, Bh both have inner interface terms, Jh only applies on \partial \Omega x (0, T) and Kh applies on \partial \Omega x T.
		for (const auto& pval : this->m_Mesh.m_SurfaceList) {
			const auto& surf_id = pval.first;
			const auto& surf_data = pval.second;
			switch (surf_data.type) {
			case SurfaceType_t::Inner:
			{
				const auto start_offset_u = surf_data.adjacent_elements[1] * num_basis;
				const auto start_offset_v = surf_data.adjacent_elements[0] * num_basis;
				for (auto bi = basis_und_t{ 0 }; bi < num_basis; ++bi) {
					for (auto bj = basis_und_t{ 0 }; bj < num_basis; ++bj) {
						const auto offset_vi = static_cast<csr_size_t>(start_offset_v + bi);
						const auto offset_uj = static_cast<csr_size_t>(start_offset_u + bj);

						const auto form_val_Ah = this->EvaluateAh_Surface(surf_id, basis_f, static_cast<basis_index_t>(bi), static_cast<basis_index_t>(bj));
						assert(std::isfinite(form_val_Ah));
#ifndef SYMMETRIC_ASSEMBLY
						matassembler(block_size + offset_vi, offset_uj) += form_val_Ah;
						matassembler(block_size + offset_uj, offset_vi) += form_val_Ah;
#endif
						matassembler(offset_vi, block_size + offset_uj) += form_val_Ah;
						matassembler(offset_uj, block_size + offset_vi) += form_val_Ah;
					}
				}

				if (!surf_data.is_time_orthogonal) {
					const auto start_offset_up = surf_data.get_upstream_element() * num_basis;
					const auto start_offset_down = surf_data.get_downstream_element() * num_basis;

					for (auto bi = basis_und_t{ 0 }; bi < num_basis; ++bi) {
						for (auto bj = basis_und_t{ 0 }; bj < num_basis; ++bj) {
							const auto form_val_Bh = this->EvaluateBh_Surface(surf_id, basis_f, static_cast<basis_index_t>(bi), static_cast<basis_index_t>(bj));
							assert(std::isfinite(form_val_Bh));
							const auto offset_upi = static_cast<csr_size_t>(start_offset_up + bj);
							const auto offset_downi = static_cast<csr_size_t>(start_offset_down + bi);
#ifndef SYMMETRIC_ASSEMBLY
							matassembler(block_size + offset_downi, offset_upi) += form_val_Bh;
#endif
							matassembler(offset_upi, block_size + offset_downi) += form_val_Bh;
						}
					}
				}
			}
			break;

			case SurfaceType_t::MidTime:
				break;

			case SurfaceType_t::EndTime:
				break;

			case SurfaceType_t::StartTime:
				break;

			case SurfaceType_t::Undefined:
				assert(false);
			}
		}
		return std::move( matassembler );
	}

	template<class BasisFuncs>
	auto AssembleMatrix_Inner() const {
		// In a dG approach, we have a given amount of functions ( BasisFuncs' size ) per element
		// Hence, the number of *active* elements in the mesh times the BasisFuncs is what we're looking for.

		const auto basis_f = BasisFuncs{};
		using basis_index_t = typename BasisFuncs::index_t;
		using basis_und_t = std::underlying_type_t<basis_index_t>;
		const auto num_elems = this->m_Mesh.m_ElementList.size();
		const auto num_basis = basis_f.size();
		using csr_size_t = typename Utility::CSRMatrixAssembler<T>::size_type;
		const auto block_size = static_cast<csr_size_t>(num_basis * num_elems);
		const auto matrix_dim = static_cast<csr_size_t>(2 * block_size);

		auto matassembler = AssembleMatrix_Base<BasisFuncs>();

		// We first sum Ah + Bh on the inner-element interfaces up
		for (auto i = ElementId_t{ 0 }; i < num_elems; ++i) {
			const auto start_offset = i * num_basis;
			for (auto bi = basis_und_t{ 0 }; bi < num_basis; ++bi) {
				for (auto bj = bi; bj < num_basis; ++bj) {
					const auto offset_vi = static_cast<csr_size_t>(start_offset + bi);
					const auto offset_uj = static_cast<csr_size_t>(start_offset + bj);

					const auto form_val_Jh = this->EvaluateJh_Inner_Element(i, basis_f, static_cast<basis_index_t>(bi), static_cast<basis_index_t>(bj));
					assert(std::isfinite(form_val_Jh));
					matassembler(block_size + offset_vi, block_size + offset_uj) += form_val_Jh;
#ifndef SYMMETRIC_ASSEMBLY
					if (bi != bj)
						matassembler(block_size + offset_uj, block_size + offset_vi) += form_val_Jh;
#endif
				}
			}
		}

		// Aside from these per element integrals, we have some interface ones:
		// Ah, Bh both have inner interface terms, Jh only applies on \partial \Omega x (0, T) and Kh applies on \partial \Omega x T.
		for (const auto& pval : this->m_Mesh.m_SurfaceList) {
			const auto& surf_id = pval.first;
			const auto& surf_data = pval.second;
			switch (surf_data.type) {
			case SurfaceType_t::Inner:
				break;

			case SurfaceType_t::MidTime:
			{
				const auto start_offset = surf_data.adjacent_elements[0] * num_basis;
				for (auto bi = basis_und_t{ 0 }; bi < num_basis; ++bi) {
					for (auto bj = bi; bj < num_basis; ++bj) {
						const auto offset_vi = static_cast<csr_size_t>(start_offset + bi);
						const auto offset_uj = static_cast<csr_size_t>(start_offset + bj);

						const auto form_val_Kh = this->EvaluateKh_Inner_Surface(surf_id, basis_f, static_cast<basis_index_t>(bi), static_cast<basis_index_t>(bj));
						assert(std::isfinite(form_val_Kh));
						matassembler(offset_vi, offset_uj) += form_val_Kh;
#ifndef SYMMETRIC_ASSEMBLY
						if(bi != bj)
							matassembler(offset_uj, offset_vi) += form_val_Kh;
#endif
					}
				}
			}
			break;

			case SurfaceType_t::EndTime:
				break;

			case SurfaceType_t::StartTime:
				break;

			case SurfaceType_t::Undefined:
				assert(false);
			}
		}
		return matassembler.AssembleMatrix(1e-13);
	}

	template<class BasisFuncs>
	auto AssembleMatrix_Boundary() const {
		// In a dG approach, we have a given amount of functions ( BasisFuncs' size ) per element
		// Hence, the number of *active* elements in the mesh times the BasisFuncs is what we're looking for.

		const auto basis_f = BasisFuncs{};
		using basis_index_t = typename BasisFuncs::index_t;
		using basis_und_t = std::underlying_type_t<basis_index_t>;
		const auto num_elems = this->m_Mesh.m_ElementList.size();
		const auto num_basis = basis_f.size();
		using csr_size_t = typename Utility::CSRMatrixAssembler<T>::size_type;
		const auto block_size = static_cast<csr_size_t>(num_basis * num_elems);
		const auto matrix_dim = static_cast<csr_size_t>(2 * block_size);

		auto matassembler = AssembleMatrix_Base<BasisFuncs>();

		// Aside from these per element integrals, we have some interface ones:
		// Ah, Bh both have inner interface terms, Jh only applies on \partial \Omega x (0, T) and Kh applies on \partial \Omega x T.
		for(const auto& pval : this->m_Mesh.m_SurfaceList) {
			const auto& surf_id = pval.first;
			const auto& surf_data = pval.second;
			switch(surf_data.type) {
				case SurfaceType_t::Inner:
					break;

				case SurfaceType_t::MidTime:
					{
						const auto start_offset = surf_data.adjacent_elements[0] * num_basis;
						for(auto bi = basis_und_t{0}; bi < num_basis; ++bi) {
							for(auto bj = bi; bj < num_basis; ++bj) {
								const auto offset_vi = static_cast<csr_size_t>(start_offset + bi);
								const auto offset_uj = static_cast<csr_size_t>(start_offset + bj);

								const auto form_val_Jh = this->EvaluateJh_Surface(surf_id, basis_f, static_cast<basis_index_t>(bi), static_cast<basis_index_t>(bj));
								assert(std::isfinite(form_val_Jh));
								matassembler(block_size + offset_vi, block_size + offset_uj) += form_val_Jh;
#ifndef SYMMETRIC_ASSEMBLY
								if (bi != bj)
									matassembler(block_size + offset_uj, block_size + offset_vi) += form_val_Jh;
#endif
							}
						}
					}
					break;

				case SurfaceType_t::EndTime:
					{
						const auto start_offset = surf_data.adjacent_elements[0] * num_basis;
						for(auto bi = basis_und_t{0}; bi < num_basis; ++bi) {
							for(auto bj = bi; bj < num_basis; ++bj) {
								const auto offset_vi = static_cast<csr_size_t>(start_offset + bi);
								const auto offset_uj = static_cast<csr_size_t>(start_offset + bj);

								const auto form_val_Kh = this->EvaluateKh_Surface(surf_id, basis_f, static_cast<basis_index_t>(bi), static_cast<basis_index_t>(bj));
								assert(std::isfinite(form_val_Kh));
								matassembler(offset_vi, offset_uj) += form_val_Kh;
#ifndef SYMMETRIC_ASSEMBLY
								if (bi != bj)
									matassembler(offset_uj, offset_vi) += form_val_Kh; 
#endif
							}
						}
					}
					break;

				case SurfaceType_t::StartTime:
					break;

				case SurfaceType_t::Undefined:
					assert(false);
			}
		}

		return matassembler.AssembleMatrix(1e-13);
	}

	template<class BasisFuncs>
	auto AssembleMatrix_Symmetric() const {
		// In a dG approach, we have a given amount of functions ( BasisFuncs' size ) per element
		// Hence, the number of *active* elements in the mesh times the BasisFuncs is what we're looking for.

		const auto basis_f = BasisFuncs{};
		using basis_index_t = typename BasisFuncs::index_t;
		using basis_und_t = std::underlying_type_t<basis_index_t>;
		const auto num_elems = this->m_Mesh.m_ElementList.size();
		const auto num_basis = basis_f.size();
		using csr_size_t = typename Utility::CSRMatrixAssembler<T>::size_type;
		const auto block_size = static_cast<csr_size_t>(num_basis * num_elems);
		const auto matrix_dim = static_cast<csr_size_t>(2 * block_size);

		auto matassembler = AssembleMatrix_Base<BasisFuncs>();

		// We first sum Ah + Bh on the inner-element interfaces up
		for (auto i = ElementId_t{ 0 }; i < num_elems; ++i) {
			const auto start_offset = i * num_basis;
			for (auto bi = basis_und_t{ 0 }; bi < num_basis; ++bi) {
				for (auto bj = bi; bj < num_basis; ++bj) {
					const auto offset_vi = static_cast<csr_size_t>(start_offset + bi);
					const auto offset_uj = static_cast<csr_size_t>(start_offset + bj);

					const auto form_val_Jh = this->EvaluateJh_Inner_Element(i, basis_f, static_cast<basis_index_t>(bi), static_cast<basis_index_t>(bj));
					assert(std::isfinite(form_val_Jh));
					matassembler(block_size + offset_vi, block_size + offset_uj) += form_val_Jh;
#ifndef SYMMETRIC_ASSEMBLY
					if (bi != bj)
						matassembler(block_size + offset_uj, block_size + offset_vi) += form_val_Jh;
#endif

					const auto form_val_Kh = this->EvaluateKh_Symmetric_Element(i, basis_f, static_cast<basis_index_t>(bi), static_cast<basis_index_t>(bj));
					assert(std::isfinite(form_val_Kh));
					matassembler(offset_vi, offset_uj) += form_val_Kh;
#ifndef SYMMETRIC_ASSEMBLY
					if (bi != bj)
						matassembler(offset_uj, offset_vi) += form_val_Kh;
#endif
				}
			}
		}

		// We need to ensure homogenous Dirichlet boundary conditions
		for (const auto& pval : this->m_Mesh.m_SurfaceList) {
			const auto& surf_id = pval.first;
			const auto& surf_data = pval.second;
			switch (surf_data.type) {
			case SurfaceType_t::Inner:
				break;

			case SurfaceType_t::MidTime:
			{
				const auto start_offset = surf_data.adjacent_elements[0] * num_basis;
				for (auto bi = basis_und_t{ 0 }; bi < num_basis; ++bi) {
					const auto offset_vi = static_cast<csr_size_t>(start_offset + bi);

					const auto form_val_L2surf = this->EvaluateJh_Surface(surf_id, basis_f, static_cast<basis_index_t>(bi), static_cast<basis_index_t>(bi));
					assert(std::isfinite(form_val_L2surf));
					if (form_val_L2surf > 5 * std::numeric_limits<T>::epsilon()) {
						matassembler.ResetRow(offset_vi);
						matassembler.ResetRow(block_size + offset_vi);

						// If the assembly is symmetric, we have to null the entire column, too
						// Otherwise there would be non-zero entries in the row
#ifdef EXACT_PRECISION_DIRICHLET
						matassembler.ResetColumn(offset_vi, true);
						matassembler.ResetColumn(block_size + offset_vi, true);

						matassembler(offset_vi, offset_vi) = T{ 1 };
						matassembler(block_size + offset_vi, block_size + offset_vi) = T{ 1 };
#else
						matassembler(offset_vi, offset_vi) = T{ 1e+20 };
						matassembler(block_size + offset_vi, block_size + offset_vi) = T{ 1e+20 };
#endif
					}
				}
			}
			break;

			case SurfaceType_t::EndTime:
				break;

			case SurfaceType_t::StartTime:
				break;

			case SurfaceType_t::Undefined:
				assert(false);
			}
		}

		return matassembler.AssembleMatrix(1e-13);
	}
};

#ifdef HAVE_MKL
template<class T, class BasisFuncs>
class STMSolver {
protected:
	const Utility::CSRMatrix<T>& m_A;
	const std::vector<T>& m_b;
	std::vector<T> m_x;
	const TetrahedralMesh<T>& m_Mesh;
	const BasisFuncs basis_f{};
	const T m_beta;
	const T m_lambda;
public:
	using ElementId_t = typename TetrahedralMesh<T>::ElementId_t;
	using SurfaceId_t = typename TetrahedralMesh<T>::SurfaceId_t;
	using Point_t = typename TetrahedralMesh<T>::Point_t;
	using NodeId_t = typename TetrahedralMesh<T>::NodeId_t;

	STMSolver(const T beta, const T lambda, const TetrahedralMesh<T>& Mesh, const Utility::CSRMatrix<T>& A, const std::vector<T>& b) : m_Mesh{ Mesh }, m_A{ A }, m_b{ b }, m_beta{beta}, m_lambda{lambda}
	{
		m_x.resize(m_A.GetNumberOfRows());
#ifndef NDEBUG
		for (const auto bi : m_b)
			assert(std::isfinite(bi));
		for (const auto Ai : m_A.m_Entries)
			assert(std::isfinite(Ai));
#endif
#ifdef USE_GMRES
		IterativeSolvers::MKL_FGMRES(m_A, m_x, m_b, 5000, 0, T{ 0 });
#else
#ifdef SYMMETRIC_ASSEMBLY
		IterativeSolvers::MKL_PARDISO_SYM(m_A, m_x, m_b);
#else
		IterativeSolvers::MKL_PARDISO(m_A, m_x, m_b);
#endif
#endif
#ifndef NDEBUG
		for (const auto xi : m_x)
			assert(std::isfinite(xi));
#endif
	}

	auto pEvaluateElement(const ElementId_t elemid, const Point_t& p) const
	{
		const auto tetrahedr = m_Mesh.ElementIdToTetrahedron(elemid);
		const auto ref_tran = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(tetrahedr);

		return pEvaluateElement_Ref(elemid, ref_tran.InverseMap(p));
	}

	auto uEvaluateElement(const ElementId_t elemid, const Point_t& p) const
	{
		const auto tetrahedr = m_Mesh.ElementIdToTetrahedron(elemid);
		const auto ref_tran = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(tetrahedr);

		return uEvaluateElement_Ref(elemid, ref_tran.InverseMap(p));
	}

	auto yEvaluateElement(const ElementId_t elemid, const Point_t& p) const
	{
		const auto tetrahedr = m_Mesh.ElementIdToTetrahedron(elemid);
		const auto ref_tran = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(tetrahedr);

		return yEvaluateElement_Ref(elemid, ref_tran.InverseMap(p));
	}

	auto pEvaluateElement_Ref(const ElementId_t elemid, const Point_t& p) const
	{
		auto pointval = T{ 0 };

		const auto num_elems = this->m_Mesh.m_ElementList.size();
		const auto num_basis = basis_f.size();
		const auto block_size = num_basis * num_elems;

		const auto start_offset = elemid * num_basis;
		for (auto bi = 0; bi < num_basis; ++bi) {
			pointval += m_x[block_size + start_offset + bi] * basis_f(static_cast<typename BasisFuncs::index_t>(bi), p);
			assert(std::isfinite(pointval));
		}
		return pointval;
	}

	auto uEvaluateElement_Ref(const ElementId_t elemid, const Point_t& p) const
	{
		auto pointval = pEvaluateElement_Ref(elemid, p);
		return T{ -1 } *(m_beta / m_lambda) * pointval;
	}

	auto yEvaluateElement_Ref(const ElementId_t elemid, const Point_t& p) const
	{
		auto pointval = T{ 0 };

		const auto num_elems = this->m_Mesh.m_ElementList.size();
		const auto num_basis = basis_f.size();

		const auto start_offset = elemid * num_basis;
		for (auto bi = 0; bi < num_basis; ++bi) {
			pointval += m_x[start_offset + bi] * basis_f(static_cast<typename BasisFuncs::index_t>(bi), p);
			assert(std::isfinite(pointval));
		}
		return pointval;
	}

	template<class TetraQuadFm, class F>
	auto L2NormSq_Element(const TetraQuadFm& tetra_quadfm, const ElementId_t elemid, const F& rhs_func) const {
		const auto cur_tetrahedron = m_Mesh.ElementIdToTetrahedron(elemid);
		const auto ref_tran = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(cur_tetrahedron);
		const auto ref_tran_det = ref_tran.GetDeterminantAbs();

		const auto integrand = [&](const auto& sp) -> auto {
			const auto p_space = ref_tran(sp);
			const auto rhsval = rhs_func(p_space[0], p_space[1], p_space[2]);
			return rhsval * rhsval;
		};

		return tetra_quadfm(integrand) * ref_tran_det;
	}

	template<class TetraQuadFm, class F>
	auto L2Norm_SpaceTime(const F& rhs_func) const {
		const auto tetra_quadfm = TetraQuadFm{};
		auto normval = T{ 0 };
		for (auto i = ElementId_t{ 0 }; i < m_Mesh.m_ElementList.size(); ++i)
			normval += L2NormSq_Element(tetra_quadfm, i, rhs_func);
		return std::sqrt(normval);
	}

	template<class TetraQuadFm, class F>
	auto L2NormErrorSq_Element(const TetraQuadFm& tetra_quadfm, const ElementId_t elemid, const bool yError, const F& rhs_func) const {
		const auto cur_tetrahedron = m_Mesh.ElementIdToTetrahedron(elemid);
		const auto ref_tran = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(cur_tetrahedron);
		const auto ref_tran_det = ref_tran.GetDeterminantAbs();

		const auto integrand = [&](const auto& sp) -> auto {
			const auto p_space = ref_tran(sp);
			const auto rhsval = rhs_func(p_space[0], p_space[1], p_space[2]);
			const auto exval = (yError ? yEvaluateElement_Ref(elemid, sp) : uEvaluateElement_Ref(elemid, sp));
			const auto errorval = rhsval - exval;
			return errorval * errorval;
		};

		return tetra_quadfm(integrand) * ref_tran_det;
	}

	template<class TetraQuadFm, class F>
	auto L2NormError_SpaceTime(const F& rhs_func, const bool yError) const {
		const auto tetra_quadfm = TetraQuadFm{};
		auto normval = T{ 0 };
		for (auto i = ElementId_t{ 0 }; i < m_Mesh.m_ElementList.size(); ++i)
			normval += L2NormErrorSq_Element(tetra_quadfm, i, yError, rhs_func);
		return std::sqrt(normval);
	}

	void PrintToVTU(const std::string& file_name, bool EvaluateY) const
	{
		auto points = vtkSmartPointer<vtkPoints>::New();
		auto cells = vtkSmartPointer<vtkCellArray>::New();
		auto dataarr = vtkSmartPointer<vtkDoubleArray>::New();
		for (auto i = ElementId_t{ 0 }; i < m_Mesh.m_ElementList.size(); ++i) {
			const auto& curelem = m_Mesh.m_ElementList[i];

			const auto tetrahedr = m_Mesh.ElementIdToTetrahedron(i);
			const auto ref_tran = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(tetrahedr);

			auto tetra = vtkSmartPointer<vtkTetra>::New();
			for (auto j = NodeId_t{ 0 }; j < curelem.corners.size(); ++j) {
				const auto p = m_Mesh.m_NodeList[curelem.corners[j]];
				const auto p_VtkId = points->InsertNextPoint(p[0], p[1], p[2]);
				tetra->GetPointIds()->SetId(j, p_VtkId);

				auto p_Ref = ref_tran.InverseMap(p);
				dataarr->InsertNextValue((EvaluateY ? yEvaluateElement_Ref(i, p_Ref) : uEvaluateElement_Ref(i, p_Ref)));
			}

			cells->InsertNextCell(tetra);
		}

		auto usgrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
		usgrid->SetPoints(points);
		usgrid->SetCells(VTK_TETRA, cells);
		usgrid->GetPointData()->SetScalars(dataarr);
		auto usgridwriter = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
		usgridwriter->SetFileName(file_name.c_str());
		usgridwriter->SetInputData(usgrid);
		usgridwriter->Write();
	}
};
#endif

#ifndef SYMMETRIC_ASSEMBLY

template<class T, class TriangQuadFm, class TetraQuadFm>
struct HeatAssembler : public STMFormEvaluator<T, TriangQuadFm, TetraQuadFm> {
	using ElementId_t = typename STMFormEvaluator<T, TriangQuadFm, TetraQuadFm>::ElementId_t;
	using SurfaceId_t = typename STMFormEvaluator<T, TriangQuadFm, TetraQuadFm>::SurfaceId_t;
	using SurfaceType_t = typename TetrahedralMesh<T>::SurfaceType_t;
	using Point_t = typename TetrahedralMesh<T>::Point_t;

	HeatAssembler(const TetrahedralMesh<T>& mesh, const T par_sigma, const T par_alpha, const T par_beta, const T par_lambda, const T par_theta) :
		STMFormEvaluator<T, TriangQuadFm, TetraQuadFm>(mesh, par_sigma, par_alpha, par_beta, par_lambda, par_theta)
	{
	}

	template<class BasisFuncs, class F0, class FT>
	auto AssembleMatrixAndLV(const F0& y0, const FT& gN) const {
		// In a dG approach, we have a given amount of functions ( BasisFuncs' size ) per element
		// Hence, the number of *active* elements in the mesh times the BasisFuncs is what we're looking for.

		const auto basis_f = BasisFuncs{};
		using basis_index_t = typename BasisFuncs::index_t;
		using basis_und_t = std::underlying_type_t<basis_index_t>;
		const auto num_elems = this->m_Mesh.m_ElementList.size();
		const auto num_basis = basis_f.size();
		using csr_size_t = typename Utility::CSRMatrixAssembler<T>::size_type;
		const auto block_size = static_cast<csr_size_t>(num_basis * num_elems);
		const auto matrix_dim = static_cast<csr_size_t>(block_size);

		auto matassembler = Utility::CSRMatrixAssembler<T>{ matrix_dim, matrix_dim };

		auto loadvec = std::vector<T>(matrix_dim);

		// We first sum Ah + Bh on the inner-element interfaces up
		for (auto i = ElementId_t{ 0 }; i < num_elems; ++i) {
			const auto start_offset = i * num_basis;
			for (auto bi = basis_und_t{ 0 }; bi < num_basis; ++bi) {
				for (auto bj = bi; bj < num_basis; ++bj) {
					const auto offset_vi = static_cast<csr_size_t>(start_offset + bi);
					const auto offset_uj = static_cast<csr_size_t>(start_offset + bj);

					auto form_val = T{ 0 };
					const auto form_val_Ah = this->EvaluateAh_Element(i, basis_f, static_cast<basis_index_t>(bi), static_cast<basis_index_t>(bj));
					assert(std::isfinite(form_val_Ah));

					matassembler(offset_vi, offset_uj) = form_val_Ah;

					if (bi != bj)
						matassembler(offset_uj, offset_vi) = form_val_Ah;
				}

				for (auto bj = basis_und_t{ 0 }; bj < num_basis; ++bj) {
					const auto offset_vi = static_cast<csr_size_t>(start_offset + bi);
					const auto offset_uj = static_cast<csr_size_t>(start_offset + bj);

					const auto form_val_Bh = this->EvaluateBh_Element(i, basis_f, static_cast<basis_index_t>(bi), static_cast<basis_index_t>(bj));
					assert(std::isfinite(form_val_Bh));

#ifndef INVERTED_PROBLEM
					matassembler(offset_vi, offset_uj) += form_val_Bh;
#else
					matassembler(offset_uj, offset_vi) += form_val_Bh;
#endif
				}
			}
		}

		// Aside from these per element integrals, we have some interface ones:
		// Ah, Bh both have inner interface terms, Jh only applies on \partial \Omega x (0, T) and Kh applies on \partial \Omega x T.
		for (const auto& pval : this->m_Mesh.m_SurfaceList) {
			const auto& surf_id = pval.first;
			const auto& surf_data = pval.second;
			switch (surf_data.type) {
				case SurfaceType_t::Inner:
					{
						const auto start_offset_u = surf_data.adjacent_elements[1] * num_basis;
						const auto start_offset_v = surf_data.adjacent_elements[0] * num_basis;
						for (auto bi = basis_und_t{ 0 }; bi < num_basis; ++bi) {
							for (auto bj = basis_und_t{ 0 }; bj < num_basis; ++bj) {
								const auto offset_uj = static_cast<csr_size_t>(start_offset_u + bj);
								const auto offset_vi = static_cast<csr_size_t>(start_offset_v + bi);

								const auto form_val_Ah = this->EvaluateAh_Surface(surf_id, basis_f, static_cast<basis_index_t>(bi), static_cast<basis_index_t>(bj));
								assert(std::isfinite(form_val_Ah));

								matassembler(offset_vi, offset_uj) += form_val_Ah;
								matassembler(offset_uj, offset_vi) += form_val_Ah;
							}
						}

						if (!surf_data.is_time_orthogonal) {
							const auto start_offset_up = static_cast<csr_size_t>(surf_data.get_upstream_element() * num_basis);
							const auto start_offset_down = static_cast<csr_size_t>(surf_data.get_downstream_element() * num_basis);

							for (auto bi = basis_und_t{ 0 }; bi < num_basis; ++bi) {
								for (auto bj = basis_und_t{ 0 }; bj < num_basis; ++bj) {
									const auto form_val_Bh = this->EvaluateBh_Surface(surf_id, basis_f, static_cast<basis_index_t>(bi), static_cast<basis_index_t>(bj));
									assert(std::isfinite(form_val_Bh));
#ifdef INVERTED_PROBLEM
									matassembler(start_offset_up + bj, start_offset_down + bi) += form_val_Bh;
#else
									matassembler(start_offset_down + bi, start_offset_up + bj) += form_val_Bh;
#endif
								}
							}
						}
					}
					break;

				case SurfaceType_t::MidTime:
					{
						const auto start_offset = surf_data.adjacent_elements[0] * num_basis;
						for (auto bi = basis_und_t{ 0 }; bi < num_basis; ++bi) {
							const auto offset_vi = static_cast<csr_size_t>(start_offset + bi);

							const auto form_val_LV_low = this->EvaluateLV_Surface(gN, surf_id, basis_f, static_cast<basis_index_t>(bi));
							assert(std::isfinite(form_val_LV_low));
							loadvec[offset_vi] += form_val_LV_low;
						}
					}
					break;
#ifdef INVERTED_PROBLEM
				case SurfaceType_t::EndTime:
#else
				case SurfaceType_t::StartTime:
#endif
					{
						const auto start_offset = surf_data.adjacent_elements[0] * num_basis;
						for (auto bi = basis_und_t{ 0 }; bi < num_basis; ++bi) {
							const auto offset_vi = static_cast<csr_size_t>(start_offset + bi);

							const auto form_val_LV_up = this->EvaluateLV_Surface(y0, surf_id, basis_f, static_cast<basis_index_t>(bi));
							assert(std::isfinite(form_val_LV_up));
							loadvec[offset_vi] += form_val_LV_up;
						}
					}
					break;

				case SurfaceType_t::Undefined:
					assert(false);
			}
		}
		return std::make_pair(matassembler.AssembleMatrix(), loadvec);
	}
};

#endif

#endif
