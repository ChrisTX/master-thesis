#ifndef TETRAHEDRAL_MESH_HPP
#define TETRAHEDRAL_MESH_HPP

#include <array>
#include <cstddef>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <tuple>
#include <map>
//#include <unordered_map>
#include <vector>

#ifndef NDEBUG
#include <iostream>
#endif

template<typename T>
class TetrahedralMesh {
public:
	const T m_StartTime;
	const T m_EndTime;

	TetrahedralMesh(const T start_time, const T end_time) : m_StartTime{ start_time }, m_EndTime{ end_time }
	{

	}

	using Point_t = std::array<T, 3>;
	using NodeId_t = typename std::vector<Point_t>::size_type;

	using ElementDescriptor_t = std::array<NodeId_t, 4>;
	using ElementId_t = std::size_t;
	struct Element_t {
		ElementDescriptor_t corners;

		bool is_in_mesh = true;
		T h;
	};

	enum class SurfaceType_t {
		Undefined,
		Inner,
		StartTime,
		MidTime,
		EndTime
	};

	struct SurfaceId_t : public std::array<NodeId_t, 3> {
		SurfaceId_t( std::initializer_list<NodeId_t> il ) {
			assert(il.size() == 3);
			auto j = std::size_t{0};
			for( auto val : il )
				(*this)[j++] = val;

			for( auto n = std::size_t{0}; n < 2; ++n ) {
				for( auto i = std::size_t{0}; i < 2; ++i ) {
					if( (*this)[i+1] < (*this)[i] )
					       std::swap( (*this)[i+1], (*this)[i] );	
				}
			}

#ifndef NDEBUG
			for (auto i = std::size_t{ 0 }; i < 2; ++i) {
				assert((*this)[i + 1] != (*this)[i]);
			}
#endif
		}
	};

	struct SurfaceData_t {
		SurfaceType_t type = SurfaceType_t::Undefined;
		std::array<ElementId_t, 2> adjacent_elements;
		bool is_time_orthogonal;
		T h;

		const auto normal_vector_by_elemid(const ElementId_t elemid) const{
			if (elemid == adjacent_elements[0]) {
				return normal_vector_first();
			}
			else {
				assert(elemid == adjacent_elements[1]);
				return normal_vector_second();
			}
		}

		const auto& normal_vector_first() const {
			assert(type != SurfaceType_t::Undefined);
			return m_normal_vector;
		}

		auto normal_vector_second() const {
			assert(type == SurfaceType_t::Inner);
			return Point_t{ -m_normal_vector[0], -m_normal_vector[1], -m_normal_vector[2] };
		}

		auto get_upstream_element() const {
			assert(!is_time_orthogonal);
			return (m_normal_vector[2] > T{ 0 } ? adjacent_elements[0] : adjacent_elements[1]);
		}

		// We save the normal vector of the first adjacent element
		// The second one is minus this one.
		Point_t m_normal_vector;
	};

	std::vector<Point_t> m_NodeList;
	std::vector<Element_t> m_ElementList;
	/* The surface list can be built on demand from the current element list
	   It takes 4N time to construct, where N is the number of elements */
	std::map<SurfaceId_t, SurfaceData_t> m_SurfaceList;

	auto ElementToTetrahedron(const Element_t& elem) const {
		auto retval = std::array<Point_t, 4>{};
		for(auto i = 0; i < 4; ++i)
			retval[i] = m_NodeList[elem.corners[i]];
		return std::move(retval);
	}

	auto ElementIdToTetrahedron(const ElementId_t elemid) const {
		return std::move(ElementToTetrahedron(m_ElementList[elemid]));
	}

	auto SurfaceIdToTriangle(const SurfaceId_t& surf) const {
		auto retval = std::array<Point_t, 3>{};
		for(auto i = 0; i < 3; ++i)
			retval[i] = m_NodeList[surf[i]];
		return std::move(retval);
	}

	auto& SurfaceDataById(const SurfaceId_t& surfid) const {
		const auto& cur_surface_it = m_SurfaceList.find( surfid );
		assert( cur_surface_it != m_SurfaceList.end() );

		return (*cur_surface_it).second;
	}

	auto InsertNode(Point_t point_to_insert) {
		m_NodeList.push_back(std::move(point_to_insert));
		return m_NodeList.size() - 1;
	}

	auto InsertElement(ElementDescriptor_t elem_to_insert) {
#ifndef NDEBUG
		for (auto i = std::size_t{ 0 }; i < 4; ++i)
			for (auto j = std::size_t{ 0 }; j < 4; ++j)
				assert((i == j) || (elem_to_insert[j] != elem_to_insert[i]));

		for (auto i = std::size_t{ 0 }; i < m_ElementList.size(); ++i) {
			auto matching_nodes = 0;
			for (auto j = std::size_t{ 0 }; j < 4; ++j)
				for (auto k = std::size_t{ 0 }; k < 4; ++k)
					if (m_ElementList[i].corners[j] == elem_to_insert[k])
						++matching_nodes;
			assert(matching_nodes != 4);
		}
#endif

		auto new_elem = Element_t{};
		new_elem.corners = std::move(elem_to_insert);
		m_ElementList.push_back(std::move(new_elem));
		return m_ElementList.size() - 1;
	}

	auto FindOrInsertNode(Point_t NodeTarget) {
		for(auto i = std::size_t{0}; i < m_NodeList.size(); ++i)
			if( m_NodeList[i] == NodeTarget )
				return i;
		
		m_NodeList.push_back( std::move( NodeTarget ) );
		return m_NodeList.size() - 1;	
	}

	void RedRefine(ElementId_t elemid_to_refine) {
		auto& elem_to_refine = m_ElementList[elemid_to_refine];
		elem_to_refine.is_in_mesh = false;

		// Use the point naming scheme as in [Bey95]
		const auto x0 = elem_to_refine.corners[0];
		const auto x1 = elem_to_refine.corners[1];
		const auto x2 = elem_to_refine.corners[2];
		const auto x3 = elem_to_refine.corners[3];

		const auto x0_l = m_NodeList[ x0 ];
		const auto x1_l = m_NodeList[ x1 ];
		const auto x2_l = m_NodeList[ x2 ];
		const auto x3_l = m_NodeList[ x3 ];

		const auto FIMidNode = [this]( auto x, auto y ) -> auto {
			auto z = Point_t{};
			for(auto i = 0; i < 3; ++i)
				z[i] = ( x[i] + y[i] ) / T{2};
			return FindOrInsertNode(z);
		};

		const auto x01 = FIMidNode(x0_l, x1_l);
		const auto x02 = FIMidNode(x0_l, x2_l);
		const auto x03 = FIMidNode(x0_l, x3_l);
		const auto x12 = FIMidNode(x1_l, x2_l);
		const auto x13 = FIMidNode(x1_l, x3_l);
		const auto x23 = FIMidNode(x2_l, x3_l);

		InsertElement({x0, x01, x02, x03});
		InsertElement({x01, x1, x12, x13});
		InsertElement({x02, x12, x2, x23});
		InsertElement({x03, x13, x23, x3});
		InsertElement({x01, x02, x03, x13});
		InsertElement({x01, x02, x12, x13});
		InsertElement({x02, x03, x13, x23});
		InsertElement({x02, x12, x13, x23});
	}

	auto CalculateNormal(const SurfaceId_t& surfid) const {
		const auto& a = m_NodeList[surfid[0]];
		const auto& b = m_NodeList[surfid[1]];
		const auto& c = m_NodeList[surfid[2]];

		const auto ac = Point_t{a[0] - c[0], a[1] - c[1], a[2] - c[2]};
		const auto bc = Point_t{b[0] - c[0], b[1] - c[1], b[2] - c[2]};
		auto normal_vector = Point_t{};
		normal_vector[0] = ac[1] * bc[2] - ac[2] * bc[1];
		normal_vector[1] = ac[2] * bc[0] - ac[0] * bc[2];
		normal_vector[2] = ac[0] * bc[1] - ac[1] * bc[0];
		auto edist = T{0};
		for(auto i = std::size_t{0}; i < 3; ++i)
			edist += normal_vector[i] * normal_vector[i];
		edist = std::sqrt(edist);
		for(auto i = std::size_t{0}; i < 3; ++i)
			normal_vector[i] /= edist;
		return std::move( normal_vector );
	}

	void CompactElementList() {
		auto vec_copy = decltype(m_ElementList){};
		vec_copy.reserve( m_ElementList.size() );

		for(auto i = decltype(m_ElementList.size()){0}; i < m_ElementList.size(); ++i) {
			auto elem_copy = m_ElementList[i];
			if(elem_copy.is_in_mesh)
				vec_copy.push_back(std::move(elem_copy));
		}
		
		vec_copy.shrink_to_fit();
		m_ElementList = std::move(vec_copy);
	}

	bool IsInTimeBorder( const SurfaceId_t& surfid, const T time_border ) {
		for(auto i = std::size_t{0}; i < 3; ++i) {
			if( std::abs(m_NodeList[surfid[i]][2] - time_border) > 5 * std::numeric_limits<T>::epsilon() ) {
				return false;
			}
		}
		return true;
	}

	void CalculateElementH(const ElementId_t curelemid) {
		auto& curelem = m_ElementList[curelemid];
		const auto a = m_NodeList[ curelem.corners[0] ];
		const auto b = m_NodeList[ curelem.corners[1] ];
		const auto c = m_NodeList[ curelem.corners[2] ];
		const auto d = m_NodeList[ curelem.corners[3] ];

		const auto eval_dist = []( auto p1, auto p2 ) -> auto {
			auto edist = T{0};
			for(auto i = 0; i < 3; ++i)
				edist += std::pow( p1[i] - p2[i], T{2} );
			return std::sqrt(edist);
		};

		curelem.h = eval_dist(a, b);
		curelem.h = std::max( curelem.h, eval_dist(a, c) );
		curelem.h = std::max( curelem.h, eval_dist(a, d) );
		curelem.h = std::max( curelem.h, eval_dist(b, c) );
		curelem.h = std::max( curelem.h, eval_dist(b, d) );
		curelem.h = std::max( curelem.h, eval_dist(c, d) );
	}

	void CalculateAllElementH() {
		for(auto i = ElementId_t{0}; i < m_ElementList.size(); ++i)
			CalculateElementH(i);
	}

	void ConsiderElementForSurfaceList(const ElementId_t curelemid, const SurfaceId_t& surfid, const NodeId_t other_node_id) {
		// Note that because this is an unordered map, this operation might insert the surface
		auto& surf = m_SurfaceList[surfid];
		const auto& curelem = m_ElementList[curelemid]; 
		switch(surf.type) {
			case SurfaceType_t::Undefined:
				{
					// Untouched so far.
					if( IsInTimeBorder(surfid, m_EndTime) )
						surf.type = SurfaceType_t::EndTime;
					else if ( IsInTimeBorder(surfid, m_StartTime) )
						surf.type = SurfaceType_t::StartTime;
					else
						surf.type = SurfaceType_t::MidTime;

					surf.adjacent_elements[0] = curelemid;
					surf.h = curelem.h;

					auto surfnm = CalculateNormal(surfid);

					const auto surf_node = m_NodeList[surfid[0]];
					const auto other_node = m_NodeList[other_node_id];
					auto scalval = T{0};
					for(auto i = std::size_t{0}; i < 3; ++i)
						scalval += ( other_node[i] - surf_node[i] ) * surfnm[i];
					surf.is_time_orthogonal = (std::abs(surfnm[2]) < 5 * std::numeric_limits<T>::epsilon());

					assert(std::abs(scalval) >= 5 * std::numeric_limits<T>::epsilon());
					if (scalval < 0)
						surf.m_normal_vector = std::move(surfnm);
					else
						surf.m_normal_vector = Point_t{ -surfnm[0], -surfnm[1], -surfnm[2] };
				}
				break;
			case SurfaceType_t::MidTime:
				// Seen once, make inner.
				surf.type = SurfaceType_t::Inner;
				surf.adjacent_elements[1] = curelemid;
				surf.h += curelem.h;
				surf.h /= T{2};
				break;
			case SurfaceType_t::EndTime:
			case SurfaceType_t::Inner:
			case SurfaceType_t::StartTime:
				// Start, End time elements or inner. Error.
				throw std::logic_error{ "Reconsidering finalized surface" };
				break;
		}
	}

	void BuildSurfaceList() {
		m_SurfaceList.clear();

		/* 	If all elements were inner ones, we'd attain a size of (4N)/2.
			Hence, 2N is a reasonable reserve for a low effective load factor */
			//m_SurfaceList.reserve( 2 * m_ElementList.size() );
		for (auto i = ElementId_t{ 0 }; i < m_ElementList.size(); ++i) {
			const auto& curelem = m_ElementList[i];
			const auto a = curelem.corners[0];
			const auto b = curelem.corners[1];
			const auto c = curelem.corners[2];
			const auto d = curelem.corners[3];
			ConsiderElementForSurfaceList(i, { a, b, c }, d);
			ConsiderElementForSurfaceList(i, { a, b, d }, c);
			ConsiderElementForSurfaceList(i, { b, c, d }, a);
			ConsiderElementForSurfaceList(i, { a, c, d }, b);
		}
#ifndef NDEBUG
		for (auto& surf_p : m_SurfaceList) {
			assert(surf_p.second.type != SurfaceType_t::MidTime || surf_p.second.is_time_orthogonal);
		}
#endif
	}

	void SplitPrism(const NodeId_t al, const NodeId_t bl, const NodeId_t cl, const NodeId_t au, const NodeId_t bu, const NodeId_t cu)
	{
		InsertElement({al, bl, cl, au});
		InsertElement({au, bu, cu, bl});
		InsertElement({bl, cl, au, cu});
	}

	void InsertFullTimePrism(const NodeId_t al, const NodeId_t bl, const NodeId_t cl)
	{
		assert(al != bl && al != cl && bl != cl);
		const auto al_node = m_NodeList[al];
		const auto bl_node = m_NodeList[bl];
		const auto cl_node = m_NodeList[cl];
		const auto au = FindOrInsertNode({al_node[0], al_node[1], m_EndTime});
		const auto bu = FindOrInsertNode({bl_node[0], bl_node[1], m_EndTime});
		const auto cu = FindOrInsertNode({cl_node[0], cl_node[1], m_EndTime});
		assert(au != bu && au != cu && bu != cu);

		SplitPrism(al, bl, cl, au, bu, cu);
	}

public:
	void UniformRefine() {
		auto last_old_elem = m_ElementList.size();
		for(auto i = ElementId_t{0}; i < last_old_elem; ++i)
			RedRefine(i);

		UpdateMesh();
	}

	void UpdateMesh() {
		CompactElementList();
		CalculateAllElementH();
		BuildSurfaceList();
	}
};

#endif
