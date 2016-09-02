#include <cassert>
#include "TetrahedralMesh.hpp"

template<class T, class TriangQuadFm, class TetraQuadFm>
class STMFormEvaluator {
protected:
	const TetrahedralMesh<T>& m_Mesh;
	const T sigma;
	const T alpha;
	const T beta;
	const T lambda;

	using ElementId_t = typename TetrahedralMesh<T>::ElementId_t;
	using SurfaceId_t = typename TetrahedralMesh<T>::SurfaceId_t;

	const auto x_scal_eval = [](const auto& u, const auto& v) -> auto {
		return u[0] * v[0] + u[1] * v[1];
	};

	const TriangQuadFm triang_quadfm{};
	const TetraQuadFm tetra_quadfm{};

public:

	STMFormEvaluator(const TetrahedralMesh<T>& mesh, const T par_sigma, const T par_alpha, const T par_beta, const T par_lambda) :
		m_Mesh{ mesh }, sigma{ par_sigma }, alpha{ par_alpha }, beta{ par_beta }, lambda{ par_lambda }
	{
	}

	template<class BasisFuncs, class BasisIndex>
	auto EvaluateAh_Element(const ElementId_t elemid, const BasisFuncs& bf, const BasisIndex biu, const BasisIndex biv) const {
		const auto cur_tetrahedron = m_Mesh.ElementIdToTetrahedron(elemid);
		const auto ref_tran = QuadratureFormulas::Tetrahedrons::ReferenceTransform(cur_tetrahedron);
		const auto ref_tran_det = ref_tran.GetDeterminant();

		const auto integrand = [&](const auto& p) -> auto {
			const auto du_org = bf.EvaluateDerivative(biu, p[0], p[1], p[2]);
			const auto du = ref_tran.TransformDerivative_Scalar(du_org);
			const auto dv_org = bf.EvaluateDerivative(biv, p[0], p[1], p[2]);
			const auto dv = ref_tran.TransformDerivative_Scalar(dv_org);

			return x_scal_eval(du, dv); 
		};

		auto intval = tetra_quadfm(integrand) / ref_tran_det;

		// At this point, first_intval is the part of the bilinear form that isspace integral.
		// We need to add the three edge terms now
		const auto first_integrand_part = [&](const auto& surfdata) -> auto {
			const auto& surf_nm = surfdata.normal_vector;

			const auto cur_triangle = m_Mesh.SurfaceIdToTriangle(elemid);
			const auto ref_triang_tran = QuadratureFormulas::Triangles::ReferenceTransform(cur_triangle);

			const auto integrand_fn = [&](const auto& sp) -> auto {
				const auto p_space = ref_triang_tran(sp);
				const auto p = ref_tran.InverseMap(p_space); 

				const auto uh = bf(biu, p[0], p[1], p[2]);
				const auto vh = bf(biv, p[0], p[1], p[2]);
				const auto du_org = bf.EvaluateDerivative(biu, p[0], p[1], p[2]);
				const auto du = ref_tran.TransformDerivative_Scalar(du_org);
				const auto dv_org = bf.EvaluateDerivative(biv, p[0], p[1], p[2]);
				const auto dv = ref_tran.TransformDerivative_Scalar(dv_org);

				const auto dux = x_scal_eval(du, surf_nm);
				const auto dvx = x_scal_eval(dv, surf_nm);

				// First two interface integral terms
				const auto f1 = T{-1} * ( (T{1}/T{2}) * ( dux * vh + uh * dvx ) );
				// Third penalty term
				const auto f2 = ( sigma/surfdata.h ) * ( uh * vh * x_scal_eval(surf_nm, surf_nm) );
				return f1 + f2;
			};

			const auto ref_triang_tran_det = ref_triang_tran.GetDeterminantSqrt();
			return triang_quadfm(integrand_fn) / ref_triang_tran_det;
		};

		const auto& curelem = m_Mesh.m_ElementList[i];
		const auto a = curelem.corners[0];
		const auto b = curelem.corners[1];
		const auto c = curelem.corners[2];
		const auto d = curelem.corners[3];

		for(const auto& si : { SurfaceId_t{a, b, c}, SurfaceId_t{a, b, d}, SurfaceId_t{b, c, d}, SurfaceId_t{a, c, d} }) {
			const auto& surfdata = m_Mesh.SurfaceDataById(si);
			if( surfdata.type == TetrahedralMesh<T>::SurfaceType_t::Inner )
				intval += first_integrand_part( surfdata );
		}
		return intval;
	}

	template<class BasisFuncs, class BasisIndex>
	auto EvaluateAh_Surface(const SurfaceId_t& surfid, const BasisFuncs& bf, const BasisIndex biu, const BasisIndex biv) const {
		// We assume that uh is an element of the first tetrahedron of the surface
		// and that vh is an element of the second tetrahedron, respectively.
		const auto& cur_surface_data = m_Mesh.SurfaceDataById(surfid);
		assert( cur_surface_data.type == TetrahedralMesh<T>::SurfaceType_t::Inner );

		const auto& elemid1 = cur_surface_data.adjacent_elements[0];
		const auto& elemid2 = cur_surface_data.adjacent_elements[1];
		const auto& tetrahedr1 = m_Mesh.ElementIdToTetrahedron(elemid1);
		const auto& ref_tran1 = QuadratureFormulas::Tetrahedrons::ReferenceTransform(tetrahedr1);
		const auto& tetrahedr2 = m_Mesh.ElementIdToTetrahedron(elemid2);
		const auto& ref_tran2 = QuadratureFormulas::Tetrahedrons::ReferenceTransform(tetrahedr2);
		const auto& surf_nm = cur_surface.normal_vector;

		const auto cur_triangle = m_Mesh.SurfaceIdToTriangle(elemid);
		const auto ref_triang_tran = QuadratureFormulas::Triangles::ReferenceTransform(cur_triangle);
		const auto ref_triang_tran_det = ref_triang_tran.GetDeterminantSqrt();

		const auto integrand_fn = [&](const auto& sp) -> auto {
			const auto p_space = ref_triang_tran(sp);
			const auto p1 = ref_tran1.InverseMap(p_space); 
			const auto p2 = ref_tran2.InverseMap(p_space); 

			const auto uh = bf(biu, p1[0], p1[1], p1[2]);
			const auto vh = bf(biv, p2[0], p2[1], p2[2]);
			const auto du_org = bf.EvaluateDerivative(biu, p1[0], p1[1], p1[2]);
			const auto du = ref_tran1.TransformDerivative_Scalar(du_org);
			const auto dv_org = bf.EvaluateDerivative(biv, p2[0], p2[1], p2[2]);
			const auto dv = ref_tran2.TransformDerivative_Scalar(dv_org);

			const auto dux = x_scal_eval(du, surf_nm);
			const auto dvx = x_scal_eval(dv, surf_nm);

				// First two interface integral terms
			const auto f1 = T{-1} * ( (T{1}/T{2}) * ( dux * vh + uh * dvx ) );
			// Third penalty term
			const auto f2 = ( sigma/surfdata.h ) * ( uh * vh * x_scal_eval(surf_nm, surf_nm) );
			return f1 + f2;
		};

		return triang_quadfm(integrand_fn) / ref_triang_tran_det;
	}

	template<class BasisFuncs, class BasisIndex>
	auto EvaluateBh_Element(const ElementId_t elemid, const BasisFuncs& bf, const BasisIndex biu, const BasisIndex biv) const {
		const auto cur_tetrahedron = m_Mesh.ElementIdToTetrahedron(elemid);
		const auto ref_tran = QuadratureFormulas::Tetrahedrons::ReferenceTransform(cur_tetrahedron);
		const auto ref_tran_det = ref_tran.GetDeterminant();

		const auto integrand = [&](const auto& p) -> auto {
			const auto uh = bf(biu, p[0], p[1], p[2]);
			const auto dv_org = bf.EvaluateDerivative(biv, p[0], p[1], p[2]);
			const auto dv = ref_tran.TransformDerivative_Scalar(dv_org);

			return uh * dv[2]; 
		};

		auto intval = tetra_quadfm(integrand) / ref_tran_det;

		// At this point, first_intval is the part of the bilinear form that isspace integral.
		// We need to add the three edge terms now
		const auto surface_integral_part = [&](const auto& surfdata) -> auto {
			const auto cur_triangle = m_Mesh.SurfaceIdToTriangle(elemid);
			const auto ref_triang_tran = QuadratureFormulas::Triangles::ReferenceTransform(cur_triangle);

			const auto integrand_fn = [&](auto sp) -> auto {
				const auto p_space = ref_triang_tran(sp);
				const auto p = ref_tran.InverseMap(p_space); 

				const auto uh = bf(biu, p[0], p[1], p[2]);
				const auto vh = bf(biv, p[0], p[1], p[2]);

				return uh * vh;
			};

			const auto ref_triang_tran_det = ref_tran.GetDeterminantSqrt();
			return triang_quadfm(integrand_fn) / ref_triang_tran_det;
		};

		const auto inner_integral_part = [&](const auto& surfdata) -> auto {
			// Given our piecewise polynomial approach, {uh}^up = uh if the time normal is > 0 and 0 otherwise
			if( surfdata.is_time_orthogonal || surfdata.adjacent_elements[surfdata.top_element] != elemid )
				return T{0};

			const auto& surf_nm = surfdata.normal_vector;

			const auto cur_triangle = m_Mesh.SurfaceIdToTriangle(elemid);
			const auto ref_triang_tran = QuadratureFormulas::Triangles::ReferenceTransform(cur_triangle);

			const auto integrand_fn = [&](auto sp) -> auto {
				const auto p_space = ref_triang_tran(sp);
				const auto p = ref_tran.InverseMap(p_space); 

				const auto uh = bf(biu, p[0], p[1], p[2]);
				const auto vh = bf(biv, p[0], p[1], p[2]);

				return uh * vh * surf_nm[2];
			};

			const auto ref_triang_tran_det = ref_triang_tran.GetDeterminantSqrt();
			return triang_quadfm(integrand_fn) / ref_triang_tran_det;
		};

		const auto& curelem = m_Mesh.m_ElementList[i];
		const auto a = curelem.corners[0];
		const auto b = curelem.corners[1];
		const auto c = curelem.corners[2];
		const auto d = curelem.corners[3];

		for(const auto& si : { SurfaceId_t{a, b, c}, SurfaceId_t{a, b, d}, SurfaceId_t{b, c, d}, SurfaceId_t{a, c, d} }) {
			const auto& surfdata = m_Mesh.SurfaceDataById(si);
			if( surfdata.type == TetrahedralMesh<T>::SurfaceType_t::EndTime )
				intval += surface_integral_part( surfdata );
			else if( surfdata.type == TetrahedralMesh<T>::SurfaceType_t::MidTime )
				intval += alpha * surface_integral_part( surfdata );
			else if( surfdata.type == TetrahedralMesh<T>::SurfaceType_t::Inner )
				intval += alpha * inner_integral_part( surfdata );
			else
				assert(false);
		}
		return intval;
	}

	template<class BasisFuncs, class BasisIndex>
	auto EvaluateBh_Surface(const SurfaceId_t& surfid, const BasisFuncs& bf, const BasisIndex biu, const BasisIndex biv) const {
		// We assume that uh is an element of the first tetrahedron of the surface
		// and that vh is an element of the second tetrahedron, respectively.
		const auto& cur_surface_data = m_Mesh.SurfaceDataById(surfid);
		assert( cur_surface_data.type == TetrahedralMesh<T>::SurfaceType_t::Inner );

		const auto& elemid1 = cur_surface_data.adjacent_elements[0];
		// If the first element (in which uh lives) is not the top element, {uh}^up will be zero for all basis functions so we can stop here.
		if( cur_surface_data.is_time_orthogonal || cur_surface_data.top_element != 0 )
			return T{0};

		const auto& elemid2 = cur_surface_data.adjacent_elements[1];
		const auto& tetrahedr1 = m_Mesh.ElementIdToTetrahedron(elemid1);
		const auto& ref_tran1 = QuadratureFormulas::Tetrahedrons::ReferenceTransform(tetrahedr1);
		const auto& tetrahedr2 = m_Mesh.ElementIdToTetrahedron(elemid2);
		const auto& ref_tran2 = QuadratureFormulas::Tetrahedrons::ReferenceTransform(tetrahedr2);
		const auto& surf_nm = cur_surface.normal_vector;

		const auto cur_triangle = m_Mesh.SurfaceIdToTriangle(elemid);
		const auto ref_triang_tran = QuadratureFormulas::Triangles::ReferenceTransform(cur_triangle);
		const auto ref_triang_tran_det = ref_triang_tran.GetDeterminantSqrt();

		const auto integrand_fn = [&](const auto& sp) -> auto {
			const auto p_space = ref_triang_tran(sp);
			const auto p1 = ref_tran1.InverseMap(p_space); 
			const auto p2 = ref_tran2.InverseMap(p_space); 

			const auto uh = bf(biu, p1[0], p1[1], p1[2]);
			const auto vh = bf(biv, p2[0], p2[1], p2[2]);
			
			return uh * vh * surf_nm[2];
		};

		return triang_quadfm(integrand_fn) / ref_triang_tran_det;
	}

	template<class BasisFuncs, class BasisIndex>
	auto EvaluateGh_Surface(const SurfaceId_t& surfid, const BasisFuncs& bf, const BasisIndex biu, const BasisIndex biv) const {
		const auto& cur_surface_data = m_Mesh.SurfaceDataById(surfid);
		const auto adj_elem_id = cur_surface_data.adjacent_elements[0];
		const auto cur_tetrahedron = m_Mesh.ElementIdToTetrahedron(adj_elem_id);
		const auto ref_tran = QuadratureFormulas::Tetrahedrons::ReferenceTransform(cur_tetrahedron);

		// At this point, first_intval is the part of the bilinear form that isspace integral.
		// We need to add the three edge terms now
		const auto surface_integral_part = [&](const auto& surfdata) -> auto {
			const auto& surf_nm = surfdata.normal_vector;

			const auto cur_triangle = m_Mesh.SurfaceIdToTriangle(surfid);
			const auto ref_triang_tran = QuadratureFormulas::Triangles::ReferenceTransform(cur_triangle);

			const auto integrand_fn = [&](auto sp) -> auto {
				const auto p_space = ref_triang_tran(sp);
				const auto p = ref_tran.InverseMap(p_space); 

				const auto uh = bf(biu, p[0], p[1], p[2]);
				const auto vh = bf(biv, p[0], p[1], p[2]);

				return uh * vh;
			};

			const auto ref_triang_tran_det = ref_triang_tran.GetDeterminantSqrt();
			return triang_quadfm(integrand_fn) / ref_triang_tran_det;
		};

		return beta * beta * surface_integral_part( surfdata ) / lambda;
	}

	template<class BasisFuncs, class BasisIndex>
	auto EvaluateHh_Surface(const SurfaceId_t& surfid, const BasisFuncs& bf, const BasisIndex biu, const BasisIndex biv) const {
		// As by assumption, the first three nodes a, b, c of elemid are the surface triangle in question
		// We assume that biu is the function of the end-time triangle and biv is of the start time triangle.
		const auto& cur_surface_data = m_Mesh.SurfaceDataById(surfid);
		assert( cur_surface_data.type == TetrahedralMesh<T>::SurfaceType_t::EndTime );
		const auto elemid_end = cur_surface_data.adjacent_elements[0];

		const auto& endtime_elem = m_Mesh.m_ElementList[elemid_end];
		assert( endtime_elem.is_in_mesh && endtime_elem.is_border_layer );
		const auto endtime_a_id = endtime_elem.corners[0];
		const auto endtime_b_id = endtime_elem.corners[1];
		const auto endtime_c_id = endtime_elem.corners[2];

		const auto endtime_a = m_Mesh.m_NodeList[endtime_a_id];
		const auto endtime_b = m_Mesh.m_NodeList[endtime_b_id];
		const auto endtime_c = m_Mesh.m_NodeList[endtime_c_id];

		assert( endtime_a[2] == endtime_b[2] && endtime_a[2] == endtime_c[2] );
		using point2d_t = std::array<T, 2>;
		const auto plain_a = point2d_t{endtime_a[0], endtime_a[1]};
		const auto plain_b = point2d_t{endtime_b[0], endtime_b[1]};
		const auto plain_c = point2d_t{endtime_c[0], endtime_c[1]};
		const auto plain_trig = std::array<point2d_t, 3>{ plain_a, plain_b, plain_c };
		// Now the plain points/trig contain the 2D triangle as part of Omega.
		const auto ref_triang_plain_tran = QuadratureFormulas::Triangles::ReferenceTransform2D(plain_trig);
		const auto ref_triang_plain_tran_det = ref_triang_plain_tran.GetDeterminantAbs();

		const auto& tetrahedr1 = m_Mesh.ElementIdToTetrahedron(elemid_end);
		const auto& ref_tran1 = QuadratureFormulas::Tetrahedrons::ReferenceTransform(tetrahedr1);

		const auto elemid_start = entime.associated_element;
		const auto& starttime_elem = m_Mesh.m_ElementList[elemid_start];
		assert( starttime_elem.is_in_mesh && starttime_elem.is_border_layer  );
		const auto& tetrahedr2 = m_Mesh.ElementIdToTetrahedron(elemid_start);
		const auto& ref_tran2 = QuadratureFormulas::Tetrahedrons::ReferenceTransform(tetrahedr2);

		const auto starttime_a = m_Mesh.m_NodeList[ starttime_elem.corners[0] ];

		const auto integrand_fn = [&](const auto& sp) -> auto {
			const auto p_2d = ref_triang_plain_tran(sp);
			const auto p_end = point_t{ p_2d[0], p_2d[1], endtime_a[2] };
			const auto p_start = point_t{ p_2d[0], p_2d[1], starttime_a[2] };
			const auto p1 = ref_tran1.InverseMap(p_end);
			const auto p2 = ref_tran2.InverseMap(p_start);

			const auto uh = bf(biu, p1[0], p1[1], p1[2]);
			const auto vh = bf(biv, p2[0], p2[1], p2[2]);
			
			return uh * vh;
		};

		return triang_quadfm(integrand_fn) / ref_triang_plain_tran_det;
	}

	template<class F, class BasisFuncs, class BasisIndex>
	auto EvaluateLV_Surface(const F& rhs_func, const SurfaceId_t& surfid, const BasisFuncs& bf, const BasisIndex biu) const {
		// As by assumption, the first three nodes a, b, c of elemid are the surface triangle in question
		// We assume that biu is the function of the end-time triangle and biv is of the start time triangle.
		const auto& cur_surface_data = m_Mesh.SurfaceDataById(surfid);
		assert( cur_surface_data.type == TetrahedralMesh<T>::SurfaceType_t::StartTime );
		const auto elemid = cur_surface_data.adjacent_elements[0];

		const auto& elem = m_Mesh.m_ElementList[elemid];
		assert( elem.is_in_mesh && elem.is_border_layer );
		const auto a_id = elem.corners[0];
		const auto b_id = elem.corners[1];
		const auto c_id = elem.corners[2];

		const auto a = m_Mesh.m_NodeList[a_id];
		const auto b = m_Mesh.m_NodeList[b_id];
		const auto c = m_Mesh.m_NodeList[c_id];

		assert( a[2] == b[2] && a[2] == c[2] );
		using point2d_t = std::array<T, 2>;
		const auto plain_a = point2d_t{a[0], a[1]};
		const auto plain_b = point2d_t{b[0], b[1]};
		const auto plain_c = point2d_t{c[0], c[1]};
		const auto plain_trig = std::array<point2d_t, 3>{ plain_a, plain_b, plain_c };
		// Now the plain points/trig contain the 2D triangle as part of Omega.
		const auto ref_triang_plain_tran = QuadratureFormulas::Triangles::ReferenceTransform2D(plain_trig);
		const auto ref_triang_plain_tran_det = ref_triang_plain_tran.GetDeterminantAbs();

		const auto& tetrahedr = m_Mesh.ElementIdToTetrahedron(elemid);
		const auto& ref_tran = QuadratureFormulas::Tetrahedrons::ReferenceTransform(tetrahedr);

		const auto integrand_fn = [&](const auto& sp) -> auto {
			const auto p_2d = ref_triang_plain_tran(sp);
			const auto p_space = point_t{ p_2d[0], p_2d[1], a[2] };
			const auto p = ref_tran.InverseMap(p_space);
			const auto uh = bf(biu, p[0], p[1], p[2]);
			const auto fx = rhs_func(p_space[0], p_space[1], p_space[2]);
			
			return uh * fx;
		};

		return triang_quadfm(integrand_fn) / ref_triang_plain_tran_det;
	}
};

template<class T, class TriangQuadFm, class TetraQuadFm>
class STMAssembler : public STMFormEvaluator<T, TriangQuadFm, TetraQuadFm> {
	template<class BasisFuncs, class F>
	auto AssembleMatrixAndLV(const F& y0, const T yOmega) const {
		// In a dG approach, we have a given amount of functions ( BasisFuncs' size ) per element
		// Hence, the number of *active* elements in the mesh times the BasisFuncs is what we're looking for.

		const auto basis_f = BasisFuncs{};
		const auto num_elems = this->m_Mesh.m_ElementList.size();
		const auto num_basis = basis_f.size();
		const auto block_size = num_basis * num_elems;
		const auto matrix_dim = 2 * block_size;

		auto matassembler = Utility::CSRMatrixAssembler<T>{ matrix_dim, matrix_dim };
		auto loadvec = std::vector<T>{matrix_dim};

		// We first sum Ah + Bh on the inner-element interfaces up
		for(auto i = ElementId_t{0}; i < num_elems; ++i) {
			const auto start_offset = i * num_basis;
			for(auto bi = 0; bi < num_basis; ++bi) {
				for(auto bj = 0; bj < num_basis; ++bj) {
					const auto form_val_AhBh = EvaluateAh_Element(i, basis_f, bi, bj) + EvaluateBh_Element(i, basis_f, bi, bj);
					matassembler(start_offset + bi, start_offset + bj) = form_val_AhBh;
					matassembler(block_size + start_offset + bi, block_size + start_offset + bj) = form_val_AhBh;
				}
			}
		}

		// Aside from these per element integrals, we have some interface ones:
		// Ah, Bh both have inner interface terms, Gh only applies on \partial \Omega x (0, T) and Hh applies on \partial \Omega x T.
		for(const auto& pval : m_Mesh.m_SurfaceList) {
			const auto& surf_id = pval.first;
			const auto& surf_data = pval.second;
			switch(surf.type) {
				case SurfaceType_t::Inner:
					{
						const auto start_offset_u = surf_data.adjacent_elements[0] * num_basis;
						const auto start_offset_v = surf_data.adjacent_elements[1] * num_basis;
						for(auto bi = 0; bi < num_basis; ++bi) {
							for(auto bj = 0; bj < num_basis; ++bj) {
								const auto form_val_AhBh = EvaluateAh_Surface(surf_id, basis_f, bi, bj) + EvaluateBh_Surface(surf_id, basis_f, bi, bj);
								matassembler(start_offset_u + bi, start_offset_v + bj) = form_val_AhBh;
								matassembler(block_size + start_offset_u + bi, block_size + start_offset_v + bj) = form_val_AhBh;
							}
						}
					}
					break;

				case SurfaceType_t::MidTime:
					{
						const auto start_offset = surf_data.adjacent_elements[0] * num_basis;
						for(auto bi = 0; bi < num_basis; ++bi) {
							for(auto bj = 0; bj < num_basis; ++bj) {
								const auto form_val_Gh = EvaluateGh_Surface(surf_id, basis_f, bi, bj);
								matassembler(block_size + start_offset + bi, start_offset + bj) = form_val_Gh;
							}
						}
					}
					break;

				case SurfaceType_t::EndTime:
					{
						const auto start_offset_u = surf_data.adjacent_elements[0] * num_basis;
						const auto other_elem = m_Mesh.m_ElementList[ surf_data.adjacent_elements[0] ].associated_element;
						const auto start_offset_v = other_elem * num_basis;
						for(auto bi = 0; bi < num_basis; ++bi) {
							for(auto bj = 0; bj < num_basis; ++bj) {
								const auto form_val_Hh = EvaluateHh_Surface(surf_id, basis_f, bi, bj);
								matassembler(start_offset_u + bi, block_size + start_offset_v + bj) = form_val_Hh;
							}
						}
					}
					break;

				case SurfaceType_t::StartTime:
					{
						const auto start_offset = surf_data.adjacent_elements[0] * num_basis;
						for(auto bi = 0; bi < num_basis; ++bi) {
							const auto form_val_LV_up = EvaluateLV_Surface(y0, surf_id, basis_f, bi, bj);
							loadvec[start_offset + bi] = form_val_LV_up;
							const auto form_val_LV_low = EvaluateLV_Surface([=yOmega]()->auto{ return yOmega; }, surf_id, basis_f, bi, bj);
							loadvec[block_size + start_offset + bi] = T{-1} * form_val_LV_low;
						}
					}
					break;

				default:
					assert(false);
			}
		}
		return std::move(matassembler);
	}
}