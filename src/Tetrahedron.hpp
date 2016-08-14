template<typename T>
struct SpacePoint {
	T x;
	T y;
	T z;

	auto& operator-=(const SpacePoint<T>& other) {
		x -= other.x;
		y -= other.y;
		z -= other.z;
		return *this;
	}

	auto& operator+=(const SpacePoint<T>& other) {
		x += other.x;
		y += other.y;
		z += other.z;
		return *this;
	}

	void L2normalize() {
		const auto norm_value = std::sqrt( x * x + y * y + z * z );
		x /= norm_value;
		y /= norm_value;
		z /= norm_value;
	}
};

template<typename T>
struct SurfacePoint {
	T x;
	T y;

	auto& operator-=(const SurfacePoint<T>& other) {
		x -= other.x;
		y -= other.y;
		return *this;
	}

	auto& operator+=(const SurfacePoint<T>& other) {
		x += other.x;
		y += other.y;
		return *this;
	}
};

template<typename T>
auto operator-(SpacePoint<T> l, const SpacePoint<T>& r) {
	return (l -= r);
}

template<typename T>
struct Triangle {
	using point_t = SurfacePoint<T>;
	point_t a;
	point_t b;
	point_t c;
};

template<typename T>
struct SpaceTriangle {
	using point_t = SpacePoint<T>;
	point_t a;
	point_t b;
	point_t c;
};

template<typename T>
struct Tetrahedron {
	using point_t = SpacePoint<T>;
	using surf_point_t = SurfacePoint<T>;
	point_t a;
	point_t b;
	point_t c;
	point_t d;
};