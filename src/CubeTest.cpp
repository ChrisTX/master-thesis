#include "TetrahedralMesh.hpp"
#include "SpaceTimeOptimizer.hpp"

#include "TetrahedralQuadrature.hpp"
#include "TriangularQuadrature.hpp"
#include "TetrahedralBasis.hpp"

#include <array>
#include <iostream>
#include <fstream>

int main() {
	auto mesh = TetrahedralMesh<double>{ 0., 1. };

	const auto al = mesh.InsertNode( { 0., 0., 0. } );
	const auto bl = mesh.InsertNode( { 0., 1., 0. } );
	const auto cl = mesh.InsertNode( { 1., 0., 0. } );
	const auto dl = mesh.InsertNode( { 1., 1., 0. } );

	mesh.InsertFullTimePrism(al, bl, cl);
	mesh.InsertFullTimePrism(bl, cl, dl);

	std::cout << "INSERT" << std::endl;
	for(auto i = 0; i < 5; ++i)
		mesh.UniformRefine();
	//mesh.UpdateMesh();

#ifdef PRINT_SURFACE_LIST
	for(auto surf_p : mesh.m_SurfaceList) {
		auto a = mesh.m_NodeList[surf_p.first[0]];
		auto b = mesh.m_NodeList[surf_p.first[1]];
		auto c = mesh.m_NodeList[surf_p.first[2]];
		auto& surf = surf_p.second;

		std::cout << "Surface between ";
		for( auto& p : {a, b, c} )
			std::cout << "( " << p[0] << ", " << p[1] << ", " << p[2] << " ) ";
		std::cout << ": type " << static_cast<int>(surf.type) << " time_ort: " << ( surf.is_time_orthogonal ? "yes" : "no" );
		if( surf.type == decltype(mesh)::SurfaceType_t::Inner )
			std::cout << " between " << surf.adjacent_elements[0] << ", " << surf.adjacent_elements[1] << " top is " << surf.top_element;
		else
			std::cout << " of " << surf.adjacent_elements[0];
		std::cout << std::endl;
	}
#endif
	auto beta = 1.;
	auto lambda = 1.;

	auto stmass = STMAssembler<double, QuadratureFormulas::Triangles::Formula_2DD2<double>, QuadratureFormulas::Tetrahedra::Formula_3DT3<double>>{ mesh, 10., 0., beta, lambda };
	auto matandlv = stmass.AssembleMatrixAndLV<BasisFunctions::TetrahedralLinearBasis<double>>( [](double, double) -> double { return 1.; }, [](double, double) -> double { return 100.; } );

#ifdef PRINT_LV
	std::ofstream lvbut("lvs.txt");
	for(auto i = std::size_t{0}; i < matandlv.second.size(); ++i)
		lvbut << matandlv.second[i] << std::endl;
	lvbut.close();
#endif

//#define PRINT_MATRIX
#ifdef PRINT_MATRIX
	std::ofstream matbut("matvals.txt");
	matbut << matandlv.first << std::endl;
	matbut.close();
#endif

#ifdef HAVE_MKL
	std::cout << "SOLVING" << std::endl;
	
	auto stmsol = STMSolver<double, BasisFunctions::TetrahedralLinearBasis<double>>{ beta, lambda, mesh, matandlv.first, matandlv.second };

	//for (auto i = std::size_t{ 0 }; i < mesh.m_ElementList.size(); ++i) {
	//	std::cout << "Element " << i << " midvalue " << stmsol.uEvaluateElement_Ref(i, {0.5, 0.5, 0.5})  << std::endl;
	//}

	stmsol.PrintToVTU("testfile-u.vtu", false);
	stmsol.PrintToVTU("testfile-y.vtu", true);
#endif
	std::cout << "DONE" << std::endl;
}
