#include "VelocityVerletIntegrator.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#include <SFML/Window.hpp>

void VelocityVerletIntegrator::validate_inputs()
{
	if (m_algorithm.get() == nullptr)
	{
		throw "Failed to run integrator. Algorithm is not set.";
	}

	if (m_renderer.get() == nullptr)
	{
		throw "Failed to run integrator. Renderer is not set.";
	}
}

void VelocityVerletIntegrator::set_algorithm(std::unique_ptr<IAlgorithmStrategy>&& algorithm)
{
	m_algorithm = std::move(algorithm);
}

void VelocityVerletIntegrator::set_renderer(std::unique_ptr<IRenderStrategy>&& renderer)
{
	m_renderer = std::move(renderer);
}

void VelocityVerletIntegrator::execute()
{
	validate_inputs();

	sf::RenderWindow window(sf::VideoMode(m_window_width, m_window_height), m_window_title);

	sf::Text frame_time;

	frame_time.setFont(m_render_font);
	frame_time.setCharacterSize(24);
	frame_time.setFillColor(sf::Color::Red);

	sf::Clock clock;

	while (window.isOpen())
	{
		sf::Event event;

		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
			{
				window.close();
			}
		}

		window.clear();

		sf::Time elapsed = clock.restart();

		sf::Clock timer;

		// run the Velocity Verlet implementation to create the vertices
		std::vector<sf::Vertex> vertices = m_algorithm.get()->run();

		// update the renderer with the vertices
		m_renderer.get()->update(vertices);

		// render the vertices
		window.draw(m_renderer.get()->get_frame());

		sf::Time duration = timer.restart();

		frame_time.setString(std::to_string(duration.asSeconds()));
		window.draw(frame_time);
		window.display();
	}
}
