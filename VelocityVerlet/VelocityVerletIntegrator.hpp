#ifndef VELOCITY_VERLET_INTEGRATOR_HPP_
#define VELOCITY_VERLET_INTEGRATOR_HPP_

#include "IAlgorithmStrategy.hpp"
#include "IRenderStrategy.hpp"

class VelocityVerletIntegrator
{
private:
	IAlgorithmStrategy& m_algorithm;
	IRenderStrategy& m_renderer;
	std::size_t m_window_width;
	std::size_t m_window_height;
	std::string m_window_title;
	sf::Font m_render_font;

	void validate_inputs();

public:
	explicit VelocityVerletIntegrator(IAlgorithmStrategy& algorithm,
		IRenderStrategy& renderer,
		std::size_t window_width,
		std::size_t window_height,
		std::string window_title,
		sf::Font render_font)
		 : m_algorithm(algorithm),
		m_renderer(renderer),
		m_window_width(window_width),
		m_window_height(window_height),
		m_window_title(window_title),
		m_render_font(render_font)
	{ }

	void set_algorithm(IAlgorithmStrategy& algorithm);
	void set_renderer(IRenderStrategy& renderer);
	void execute();
};

#endif // !VELOCITY_VERLET_INTEGRATOR_HPP_
