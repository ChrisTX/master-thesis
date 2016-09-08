#include "TetrahedralMesh.hpp"

#include <array>
#include <iostream>

int main() {
	auto mesh = TetrahedralMesh<double>{ 0., 1. };

	const auto al = mesh.InsertNode( { 0., 0., 0. } );
	const auto bl = mesh.InsertNode( { 0., 1., 0. } );
	const auto cl = mesh.InsertNode( { 1., 0., 0. } );
	const auto dl = mesh.InsertNode( { 1., 1., 0. } );

	mesh.InsertFullTimePrism(al, bl, cl);
	//mesh.InsertFullTimePrism(bl, cl, dl);

	std::cout << "INSERT" << std::endl;

	mesh.UniformRefine();

	std::cout << "DONE" << std::endl;
}
