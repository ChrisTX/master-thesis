#include "TetrahedralMesh.hpp"
#include "SpaceTimeOptimizer.hpp"

#include <array>
#include <iostream>

int main() {
	auto mesh = TetrahedralMesh<double>{ 0., 1. };

	const auto al = mesh.InsertNode( { 0., 0., 0. } );
	const auto bl = mesh.InsertNode( { 0., 1., 0. } );
	const auto cl = mesh.InsertNode( { 1., 0., 0. } );
	const auto dl = mesh.InsertNode( { 1., 1., 0. } );

	mesh.InsertFullTimePrism(al, bl, cl);
	mesh.InsertFullTimePrism(dl, bl, cl);

	std::cout << "INSERT" << std::endl;
	
	mesh.UniformRefine();
	
	/*for(auto surf_p : mesh.m_SurfaceList) {
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
	}*/ 

	auto stmsol = STMSolver<double, BasisFunctions::TetrahedralLinearBasis>;

	std::cout << "DONE" << std::endl;
}
