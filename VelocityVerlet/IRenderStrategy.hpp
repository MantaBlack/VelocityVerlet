#ifndef IRENDER_STRATEGY_HPP_
#define IRENDER_STRATEGY_HPP_

#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Drawable.hpp>
#include <vector>

class IRenderStrategy
{
public:
	virtual ~IRenderStrategy() = default;

	virtual void update(std::vector<sf::Vertex> vertices) = 0;

	virtual const sf::Drawable& get_frame() const = 0;
};

#endif // !IRENDER_STRATEGY_HPP_
