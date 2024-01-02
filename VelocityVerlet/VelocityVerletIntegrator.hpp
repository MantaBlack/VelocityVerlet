#ifndef VELOCITY_VERLET_INTEGRATOR_HPP_
#define VELOCITY_VERLET_INTEGRATOR_HPP_

#include "IAlgorithmStrategy.hpp"
#include "IRenderStrategy.hpp"

class VelocityVerletIntegrator
{
private:
	std::unique_ptr<IAlgorithmStrategy> m_algorithm;
	std::unique_ptr<IRenderStrategy> m_renderer;
	std::size_t m_window_width;
	std::size_t m_window_height;
	std::string m_window_title;
	sf::Font m_render_font;

	void validate_inputs();

public:
	explicit VelocityVerletIntegrator(std::unique_ptr<IAlgorithmStrategy> &&algorithm,
		std::unique_ptr<IRenderStrategy> &&renderer,
		std::size_t window_width,
		std::size_t window_height,
		std::string window_title,
		sf::Font render_font)
		: m_algorithm(std::move(algorithm)),
		m_renderer(std::move(renderer)),
		m_window_width(window_width),
		m_window_height(window_height),
		m_window_title(window_title),
		m_render_font(render_font)
	{ }

	void set_algorithm(std::unique_ptr<IAlgorithmStrategy>&& algorithm);
	void set_renderer(std::unique_ptr<IRenderStrategy>&& renderer);
	void execute();
};

#endif // !VELOCITY_VERLET_INTEGRATOR_HPP_
