#ifndef IALGORITHM_STRATEGY_HPP_
#define IALGORITHM_STRATEGY_HPP_

#include <SFML/Graphics.hpp>

#include <vector>

class IAlgorithmStrategy
{
public:
	virtual ~IAlgorithmStrategy() = default;

	virtual std::vector<sf::Vertex> run() = 0;
};

#endif // !IALGORITHMSTRATEGY
