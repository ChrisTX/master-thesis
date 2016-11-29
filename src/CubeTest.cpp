//#define HEAT_SYSTEM
//#define SYMMETRIC_SYSTEM
//#define INNER_SYSTEM
#define QUADRATIC_BASIS
//#define EXACT_PRECISION_DIRICHLET
//#define USE_GMRES
//#define KR_SYSTEM
//#define HOLE

#if defined(HEAT_SYSTEM) || defined(KR_SYSTEM) 
#undef SYMMETRIC_ASSEMBLY
#else
#define SYMMETRIC_ASSEMBLY
#endif

#include "TetrahedralMesh.hpp"
#include "SpaceTimeOptimizer.hpp"

#include "TetrahedralQuadrature.hpp"
#include "TriangularQuadrature.hpp"
#include "TetrahedralBasis.hpp"

#include <array>
#include <iostream>
#include <fstream>

template<class F, class T>
void print_cont_function_to_VTU(const std::string& file_name, TetrahedralMesh<T> mesh, const F& f) {
	using ElementId_t = typename TetrahedralMesh<T>::ElementId_t;
	using NodeId_t = typename TetrahedralMesh<T>::NodeId_t;
	auto points = vtkSmartPointer<vtkPoints>::New();
	auto cells = vtkSmartPointer<vtkCellArray>::New();
	auto dataarr = vtkSmartPointer<vtkDoubleArray>::New();
	for (auto i = ElementId_t{ 0 }; i < mesh.m_ElementList.size(); ++i) {
		const auto& curelem = mesh.m_ElementList[i];

		const auto tetrahedr = mesh.ElementIdToTetrahedron(i);
		const auto ref_tran = QuadratureFormulas::Tetrahedra::ReferenceTransform<T>(tetrahedr);

		auto tetra = vtkSmartPointer<vtkTetra>::New();
		for (auto j = NodeId_t{ 0 }; j < curelem.corners.size(); ++j) {
			const auto p = mesh.m_NodeList[curelem.corners[j]];
			const auto p_VtkId = points->InsertNextPoint(p[0], p[1], p[2]);
			tetra->GetPointIds()->SetId(j, p_VtkId);

			auto p_Ref = ref_tran.InverseMap(p);
			dataarr->InsertNextValue(f(p[0], p[1], p[2]));
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


int main() {

#ifdef SYMMETRIC_SYSTEM
	auto mesh = TetrahedralMesh<double>{ 0., 0.1 };

	const auto xdelta = 0.1;
	const auto xmax = 10;
	const auto ymax = 10;
#elif defined(HOLE)
	auto mesh = TetrahedralMesh<double>{ 0., 1. };

	const auto xdelta = 0.1;
	const auto xmax = 10;
	const auto ymax = 10;
#else
	auto mesh = TetrahedralMesh<double>{ 0., 1. };

	const auto xdelta = 1.;
	const auto xmax = 1;
	const auto ymax = 1;
#endif

	for (auto lx = 0; lx < xmax; ++lx) {
		for (auto ly = 0; ly < ymax; ++ly) {
#ifdef HOLE
			if (lx > 1 && lx < xmax - 2 && ly > 1 && ly < ymax - 2)
				continue;
#endif
			const auto ax = lx * xdelta;
			const auto ay = ly * xdelta;
			const auto dx = ax + xdelta;
			const auto dy = ay + xdelta;

			const auto al = mesh.FindOrInsertApproximateNode({ ax, ay, 0. });
			const auto bl = mesh.FindOrInsertApproximateNode({ ax, dy, 0. });
			const auto cl = mesh.FindOrInsertApproximateNode({ dx, ay, 0. });
			const auto dl = mesh.FindOrInsertApproximateNode({ dx, dy, 0. });

			mesh.InsertFullTimePrism(al, bl, cl);
			mesh.InsertFullTimePrism(cl, dl, bl);
		}
	}

	std::cout << "INSERT" << std::endl;

#if defined(PRINT_MATRIX)
	const auto reflim = 0;
#elif !defined(NDEBUG)
	const auto reflim = 2;
#elif defined(SYMMETRIC_SYSTEM)
	const auto reflim = 3;
#else
	const auto reflim = 4;
#endif
	for(auto i = 0; i < reflim; ++i)
		mesh.UniformRefine();
	if (!reflim)
		mesh.UpdateMesh();

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

#if !defined(NDEBUG) && !defined(NO_CHECK_GEOMETRY)
	for (auto surf_p : mesh.m_SurfaceList) {
		auto& surf = surf_p.second;
		if (surf.type == decltype(mesh)::SurfaceType_t::MidTime) {
			auto a = mesh.m_NodeList[surf_p.first[0]];
			auto b = mesh.m_NodeList[surf_p.first[1]];
			auto c = mesh.m_NodeList[surf_p.first[2]];

			if (std::abs(a[0] - b[0]) < 5 * std::numeric_limits<double>::epsilon()) {
				if (std::abs(a[0] - c[0]) < 5 * std::numeric_limits<double>::epsilon()) {
					if (std::abs(a[0]) < 5 * std::numeric_limits<double>::epsilon() || std::abs(a[0] - 1.) < 5 * std::numeric_limits<double>::epsilon()) {
						continue;
					}
				}
			}

			if (std::abs(a[1] - b[1]) < 5 * std::numeric_limits<double>::epsilon()) {
				if (std::abs(a[1] - c[1]) < 5 * std::numeric_limits<double>::epsilon()) {
					if (std::abs(a[1]) < 5 * std::numeric_limits<double>::epsilon() || std::abs(a[1] - 1.) < 5 * std::numeric_limits<double>::epsilon()) {
						continue;
					}
				}
			}

			assert(false);
		}
	}
#endif

	const auto pi = 3.14159265359;
	
	auto beta = 0.2;
#ifdef SYMMETRIC_SYSTEM
	auto lambda = std::pow(pi, -4.);
#else
	auto lambda = 0.1;
#endif
	auto alpha = 0.2;
	auto sigma = 50.;
	auto theta = 0.2;

#ifdef HEAT_SYSTEM
	auto stmass = HeatAssembler<double, QuadratureFormulas::Triangles::Formula_2DD5<double>, QuadratureFormulas::Tetrahedra::Formula_3DT3<double>>{ mesh, sigma, alpha, beta, lambda, theta };
#else
	auto stmass = STMAssembler<double, QuadratureFormulas::Triangles::Formula_2DD5<double>, QuadratureFormulas::Tetrahedra::Formula_3DT3<double>>{ mesh, sigma, alpha, beta, lambda, theta };
#endif

#ifdef QUADRATIC_BASIS
	using basis_t = BasisFunctions::TetrahedralQuadraticBasis<double>;
#else
	using basis_t = BasisFunctions::TetrahedralLinearBasis<double>;
#endif

#ifndef HEAT_SYSTEM
#ifdef INNER_SYSTEM
	auto matA = stmass.AssembleMatrix_Inner<basis_t>();
	auto lvA = stmass.AssembleLV_Inner<basis_t>([](double x, double y, double t) -> double { return 10 * t; });
#elif defined(SYMMETRIC_SYSTEM)
	const auto a = 1.;
	const auto T = 0.1;
	lambda = std::pow(pi, -4);
	beta = 1.;
	auto wa = [&](double x, double y, double t) -> double { return std::exp(a * pi * pi * t) * std::sin(pi * x) * std::sin(pi * y); };
	auto f = [&](double x, double y, double) -> double { return -1. * std::pow(pi, 4.) * wa(x, y, T); };
	auto yQ = [&](double x, double y, double t) -> double { return ( (a * a - 5.) / (2. + a) ) * pi * pi * wa(x, y, t) + 2. * pi * pi * wa(x, y, T); };
	auto y0 = [&](double x, double y, double) -> double { return ((-1.) / (2. + a)) * pi * pi * wa(x, y, 0.); };

	auto matA = stmass.AssembleMatrix_Symmetric<basis_t>();
	auto lvA = stmass.AssembleLV_Symmetric<basis_t>(f, yQ, y0);
#elif defined(KR_SYSTEM)

#else
	auto matA = stmass.AssembleMatrix_Boundary<basis_t>();
	auto lvA = stmass.AssembleLV_Boundary<basis_t>([](double, double, double) -> double { return 1.; }, [](double, double, double) -> double { return 40.; });
#endif
#else
	auto matAandLV = stmass.AssembleMatrixAndLV<basis_t>([](double, double, double) -> double { return 1.; }, [](double, double, double) -> double { return 1.; });
	auto matA = matAandLV.first;
	auto lvA = matAandLV.second;
#endif

#ifdef PRINT_LV
	std::ofstream lvbut("lvs.txt");
	for(auto i = std::size_t{0}; i < lvA.size(); ++i)
		lvbut << lvA[i] << std::endl;
	lvbut.close();
#endif

#ifdef PRINT_MATRIX
	std::ofstream matbut("matvals.txt");
	matbut << matA << std::endl;
	matbut.close();
#endif

#ifdef HAVE_MKL
	std::cout << "SOLVING" << std::endl;
	
	auto stmsol = STMSolver<double, basis_t>{ beta, lambda, mesh };
#ifdef KR_SYSTEM
	stmsol.SolveRestricted(stmass, [](double, double, double) -> double { return 1.; }, [](double, double, double) -> double { return 40.; }, 0., 40., 0.15);
#else
	stmsol.Solve(matA, lvA);
#endif

#ifndef HEAT_SYSTEM
	stmsol.PrintToVTU("testfile-u.vtu", false);
#endif
	stmsol.PrintToVTU("testfile-y.vtu", true);
#endif

	std::cout << "PRINTING" << std::endl;
	
#ifdef SYMMETRIC_SYSTEM
#define STOP_PROGRAM
	auto u_opt = [&](double x, double y, double t) -> double { return -1. * std::pow(pi, 4.) * ( wa(x, y, t) - wa(x, y, T) ); };
	auto y_opt = [&](double x, double y, double t) -> double { return ( (-1.)/(2+a) ) * std::pow(pi, 2.) * wa(x, y, t); };

	print_cont_function_to_VTU("optimum-y.vtu", mesh, y_opt);
	print_cont_function_to_VTU("optimum-u.vtu", mesh, u_opt);

	auto wa_deriv = [&](double x, double y, double t) -> auto {
		return std::array<double, 3>{
			std::exp(a * pi * pi * t) * std::cos(pi * x) * std::sin(pi * y),
			std::exp(a * pi * pi * t) * std::sin(pi * x) * std::cos(pi * y),
			a * pi * pi * std::exp(a * pi * pi * t) * std::sin(pi * x) * std::sin(pi * y)
		};
	};

	std::cout << "y error\t" << stmsol.L2NormError_SpaceTime<QuadratureFormulas::Tetrahedra::Formula_3DT3<double>>(y_opt, true) << std::endl;
	std::cout << "u error\t" << stmsol.L2NormError_SpaceTime<QuadratureFormulas::Tetrahedra::Formula_3DT3<double>>(u_opt, false) << std::endl;
#endif
	std::cout << "DONE" << std::endl;

#if !defined(NDEBUG) || defined(STOP_PROGRAM)
	std::cout << "Please press enter to continue..." << std::endl;
	std::cin.get();
#endif
}

