#include <type_traits>
#include <utility>
#include <array>

namespace QuadratureFormulas {
    template<typename P, typename K, typename F>
    using integral_res_t = decltype(std::declval<std::result_of_t<F(P)>>() * std::declval<typename K::value_type>());

	template<typename L, typename K, typename F>
	auto EvaluateQuadrature(const L& points, const K& weights, const F& f_integrand) {
        using size_type = typename K::size_type;
        static_assert(std::is_same<size_type, std::result_of_t<K.size()>(), "size_type is ill declared");

        const auto amount_of_weights = weights.size();
        auto result_value = integral_res_t<typename L::value_type K, F>{0};
        for(auto i = size_type{0}; i < amount_of_weights; ++i)
        	result_value += weights[i] * f_integrand(points[i]);
       	return result_value;
	}
}