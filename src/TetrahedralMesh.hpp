#include <array>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Tetrahedron.hpp"

template<typename T>
class TetrahedralMesh {
public:
	using Point_t = std::array<T, 3>;
	using NodeId_t = typename std::vector<Point_t>::size_type;

	using ElementDescriptor_t = std::array<NodeId_t, 4>;

	struct Element_t {
		ElementDescriptor_t corners;

		bool is_in_mesh = true;
		ElementId_t replacement_first_child;
		bool is_border_layer = false;
		ElementId_t associated_element;
		T h;
	};

	using ElementId_t = typename std::vector<Element_t>::size_type;
	using SurfaceId_t = std::array<NodeId_t, 3>;

	enum class SurfaceType_t {
		Undefined,
		Inner,
		StartTime,
		MidTime,
		EndTime
	};

	struct SurfaceData_t {
		SurfaceType_t type = SurfaceType_t::Undefined;
		std::array<ElementId_t, 2> adjacent_elements;
		bool is_time_orthogonal;
		std::size_t top_element;
		Point_t normal_vector;
		T h;
	};

	std::vector<Point_t> m_NodeList;
	std::vector<Element_t> m_ElementList;
	std::vector<ElementId_t> m_ElementBottomLayer;
	/* The surface list can be built on demand from the current element list
	   It takes 4N time to construct, where N is the number of elements */
	std::unordered_map<SurfaceId_t, SurfaceData_t> m_SurfaceList;

	T m_StartTime;
	T m_EndTime;

	auto ElementToTetrahedron(const Element_t& elem) const {
		auto retval = std::array<Point_t, 4>;
		for(auto i = 0; i < 4; ++i)
			retval[i] = m_NodeList[elem.corners[i]];
		return std::move(retval);
	}

	auto ElementIdToTetrahedron(const ElementId_t elemid) const {
		return std::move(ElementToTetrahedron(m_ElementList[elemid]));
	}

	auto SurfaceIdToTriangle(const SurfaceId_t& surf) const {
		auto retval = std::array<Point_t, 3>;
		for(auto i = 0; i < 3; ++i)
			retval[i] = m_NodeList[surf[i]];
		return std::move(retval);
	}

	auto& SurfaceDataById(const SurfaceId_t& surfid) const {
		const auto& cur_surface_it = m_Mesh.m_SurfaceList.find( surfid );
		assert( cur_surface_it != m_Mesh.m_SurfaceList.end() );

		return *cur_surface_it;
	}

	auto InsertNode(Point_t point_to_insert) {
		m_NodeList.push_back(std::move(point_to_insert));
		return m_NodeList.size() - 1;
	}

	auto InsertElement(Element_t elem_to_insert) {
		m_ElementList.push_back(std::move(elem_to_insert));
		return m_ElementList.size() - 1;
	}

	void ReplaceElement(ElementId_t elem_to_replace, ElementId_t first_child_elem) {
		auto& elem_data = m_ElementList[elem_to_replace];
		elem_data.is_in_mesh = false;
		elem_data.first_child_elem = first_child_elem;
	}

	void RedRefine(ElementId_t elemid_to_refine) {
		auto& elem_to_refine = m_ElementList[elemid_to_refine];

		// Use the point naming scheme as in [Bey95]
		const auto x0 = elem_to_refine.corners[0];
		const auto x1 = elem_to_refine.corners[1];
		const auto x2 = elem_to_refine.corners[2];
		const auto x3 = elem_to_refine.corners[3];
		const auto x01 = (x0 + x1)/T{2};
		const auto x02 = (x0 + x2)/T{2};
		const auto x03 = (x0 + x3)/T{2};
		const auto x12 = (x1 + x2)/T{2};
		const auto x13 = (x1 + x3)/T{2};
		const auto x23 = (x2 + x3)/T{2};

		const auto first_new_element_id = m_ElementList.size();

		InsertElement({x0, x01, x02, x03});
		InsertElement({x01, x1, x12, x13});
		InsertElement({x02, x12, x2, x23});
		InsertElement({x03, x13, x23, x3});
		InsertElement({x01, x02, x03, x13});
		InsertElement({x01, x02, x12, x13});
		InsertElement({x02, x03, x13, x23});
		InsertElement({x02, x12, x13, x23});

		ReplaceElement(elemid_to_refine, first_child_elem);
	}

	void UpdateNormal(const SurfaceId_t& surfid) {
		const auto& a = m_NodeList[surfid[0]];
		const auto& b = m_NodeList[surfid[1]];
		const auto& c = m_NodeList[surfid[2]];

		const auto ac = a - c;
		const auto bc = b - c;
		auto normal_vector = Point_t{};
		normal_vector[0] = ac.b * bc.c - ac.c * bc.b;
		normal_vector[1] = ac.c * bc.a - ac.a * bc.c;
		normal_vector[2] = ac.a * bc.b - ac.b * bc.a;
		normal_vector.normalize();
		m_SurfaceList[surfid].normal_vector = std::move( normal_vector );
	}

	void UpdateAssociation(Element_t old_elem) {
		const auto& old_elem_data = m_ElementList[old_elem].element_data;
		assert(old_elem_data.is_border_layer);

		// We do not remove elements from the vector, so we might have an element that was already processed
		assert(!old_elem_data.is_in_mesh);

		const auto first_child_id = old_elem_data.replacement_first_child;
		const auto other_old_elem = old_elem_data.associated_element;
		assert(other_old_elem < m_ElementList.size());
		const auto& other_old_elem_data = m_ElementList[other_old_elem].element_data;
		const auto first_other_child_id = other_old_elem_data.replacement_first_child;
		assert(!other_old_elem_data.is_in_mesh && other_old_elem_data.replacement_first_child && other_old_elem_data.associated_element == old_elem);
		assert(!other_old_elem_data.is_in_mesh && other_old_elem_data.is_border_layer);

		/* 	We assume: a, b, c correspond to each other, and d is inside. Then, x01 = x01, ... by the way of the construction.
			As of such, the elements that are in the border layers are 0, 1, 2 and 5 (all elements with only one 3 in its nodes) */
		for(auto i : {0, 1, 2, 5}) {
			const auto old_elem_child_id = first_old_child_id + i;
			const auto other_elem_child_id = first_other_child_id + i;
			auto& cur_old_elem_child = m_ElementList[old_elem_child_id];
			auto& cur_other_elem_child = m_ElementList[other_elem_child_id];
			auto& cur_child_data = cur_old_elem_child.element_data;
			auto& cur_other_child_data = cur_other_elem_child.element_data;

			cur_child_data.is_in_mesh = true;
			cur_other_child_data.is_in_mesh = true;
			cur_child_data.is_border_layer = true;
			cur_other_child_data.is_border_layer = true;

			cur_child_data.associated_element = other_elem_child_id;
			cur_other_child_data.associated_element = old_elem_child_id;
			m_ElementBottomLayer.push_back(old_elem_child_id);
			m_ElementBottomLayer.push_back(other_elem_child_id);
		}
	}

	void UpdateAllAssociations() {
		const auto oldbottomlayer = m_ElementBottomLayer;
		m_ElementBottomLayer.clear();

		for(const auto i : oldbottomlayer)
			UpdateAssociation(i);
	}

	void CompactElementList() {
		auto vec_copy = decltype(m_ElementList){};
		vec_copy.reserve( m_ElementList.size() );

		for(const auto i : m_ElementList)
			vec_copy.push_back(i);
		
		vec_copy.shrink_to_fit();
	}

	bool IsInTimeBorder( const SurfaceId_t& surfid, const T time_border ) {
		for(auto i = 0; i < 3; ++i) {
			if( std::abs(m_NodeList[surfid[i]].z - time_border) > 5 * std::numeric_limits<T>::epsilon() ) {
				return false;
			}
		}
		return true;
	}

	void CalculateElementH(const ElementId_t curelemid) {
		const auto& curelem = m_ElementList[i];
		const auto a = curelem.corners[0];
		const auto b = curelem.corners[1];
		const auto c = curelem.corners[2];
		const auto d = curelem.corners[3];

		curelem.h = std::abs(a - b);
		curelem.h = std::max( curelem.h, std::abs(a - c) );
		curelem.h = std::max( curelem.h, std::abs(a - d) );
		curelem.h = std::max( curelem.h, std::abs(b - c) );
		curelem.h = std::max( curelem.h, std::abs(b - d) );
		curelem.h = std::max( curelem.h, std::abs(c - d) );
	}

	void ConsiderElementForSurfaceList(const ElementId_t curelemid, const SurfaceId_t& surfid, const NodeId_t other_node_id) {
		// Note that because this is an unordered map, this operation might insert the surface
		auto& surf = m_SurfaceList[surfid];
		UpdateNormal(surfid);
		const auto surfnm = surf.normal_vector;
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

					const auto surf_node = m_NodeList[surfid[0]];
					const auto other_node = m_NodeList[other_node_id];
					auto scalval = T{0};
					for(auto i = 0; i < 3; ++i)
						scalval += ( other_node[i] - surf_node[i] ) * surfnm[i];
					if( scalval == T{0} ) {
						surf.is_time_orthogonal = true;
					}
					else {
						surf.top_element = ( scalval > 0 ) ? 0 : 1;
						surf.is_time_orthogonal = false;
					}
				}
				break;
			case SurfaceType_t::MidTime:
				// Seen once, make inner.
				surf.type = SurfaceType_t::Inner;
				surf.adjacent_elements[1] = curelemid;
				surf.h += curelem.h;
				surf.h /= T{2};
				break;
			default:
				// Start, End time elements or inner. Error.
				throw std::logic_error{ "Reconsidering finalized surface" };
				break;
		}
	}

	void BuildSurfaceList() {
		m_SurfaceList.clear();

		/* 	If all elements were inner ones, we'd attain a size of (4N)/2.
			Hence, 2N is a reasonable reserve for a low effective load factor */
		m_SurfaceList.reserve( 2 * m_ElementList.size() );
		for(auto i = ElementId_t{0}; i < m_ElementList.size(); ++i) {
			const auto& curelem = m_ElementList[i];
			const auto a = curelem.corners[0];
			const auto b = curelem.corners[1];
			const auto c = curelem.corners[2];
			const auto d = curelem.corners[3];
			ConsiderElementForSurfaceList(i, {a, b, c}, d);
			ConsiderElementForSurfaceList(i, {a, b, d}, c);
			ConsiderElementForSurfaceList(i, {b, c, d}, a);
			ConsiderElementForSurfaceList(i, {a, c, d}, b);
		}
	}

public:
	void UniformRefine() {
		for(auto i = ElementId_t{0}; i < m_ElementList.size(); ++i) {
			RedRefine(i);
		}
		UpdateAllAssociations();
		CompactElementList();
	}
};