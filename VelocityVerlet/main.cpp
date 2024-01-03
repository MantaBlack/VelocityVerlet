#include "VelocityVerletIntegrator.hpp"
#include "SingleThreadedVelocityVerlet.hpp"
#include "VertexBufferRenderer.hpp"

#include <iostream>
#include <random>
#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <vector>

#pragma comment(lib, "sfml-graphics-d.lib")
#pragma comment(lib, "sfml-window-d.lib")
#pragma comment(lib, "sfml-system-d.lib")
#pragma comment(lib, "freetype.lib")
#pragma comment(lib, "gdi32.lib")

static std::random_device g_random_device;
static std::mt19937 g_random_engine(g_random_device());

static std::vector<sf::Vector3f> generate_starting_positions(const std::size_t num_particles,
	const float min_val,
	const float max_val)
{
	std::vector<sf::Vector3f> positions;

	std::uniform_real_distribution<float> dist(min_val, max_val);

	for (size_t i = 0; i < num_particles; ++i)
	{
		positions.push_back
		(
			sf::Vector3f(dist(g_random_engine), dist(g_random_engine), dist(g_random_engine))
		);
	}

	return positions;
}

static std::vector<sf::Vector3f> generate_starting_velocities(const std::size_t num_particles,
	const float min_val,
	const float max_val)
{
	std::vector<sf::Vector3f> velocities;

	std::uniform_real_distribution<float> dist(min_val, max_val);

	for (size_t i = 0; i < num_particles; ++i)
	{
		velocities.push_back
		(
			sf::Vector3f(dist(g_random_engine), dist(g_random_engine), dist(g_random_engine))
		);
	}

	return velocities;
}

static std::vector<float> generate_masses(const std::size_t num_particles,
	const float min_val,
	const float max_val)
{
	std::vector<float> masses;

	std::uniform_real_distribution<float> dist(min_val, max_val);

	for (size_t i = 0; i < num_particles; ++i)
	{
		masses.push_back(dist(g_random_engine));
	}

	return masses;
}

int main()
{
	const std::size_t window_width = 1000;
	const std::size_t window_height = 1000;
	const std::string window_title = "Velocity Verlet";

	const std::size_t num_particles = 1000;
	const float time_step = 1.f;

	std::vector<sf::Vector3f> positions = generate_starting_positions(num_particles, 100.f, 800.f);
	std::vector<sf::Vector3f> velocities = generate_starting_velocities(num_particles, 1.f, 1.5f);
	std::vector<float> masses = generate_masses(num_particles, 1.f, 50.f);	

	sf::Font font;

	if (!font.loadFromFile("saxmono.ttf"))
	{
		throw std::string("Failed to load font file");
	}

	SingleThreadedVelocityVerlet algorithm(num_particles,
		time_step,
		positions,
		velocities,
		masses);

	VertexBufferRenderer renderer(sf::VertexBuffer::Stream, sf::Points);

	VelocityVerletIntegrator integrator(algorithm,
		renderer,
		window_width,
		window_height,
		window_title,
		font);

	integrator.execute();

	return 0;
}