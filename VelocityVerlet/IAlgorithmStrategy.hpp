#ifndef IALGORITHM_STRATEGY_HPP_
#define IALGORITHM_STRATEGY_HPP_

#include <SFML/Graphics.hpp>
#include <SFML/System/Vector3.hpp>
#include <vector>

class IAlgorithmStrategy
{
public:
	virtual ~IAlgorithmStrategy() = default;

	virtual std::vector<sf::Vector3f> run() const = 0;
};

#endif // !IALGORITHMSTRATEGY
