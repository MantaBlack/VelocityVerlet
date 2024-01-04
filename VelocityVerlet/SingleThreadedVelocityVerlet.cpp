#include "SingleThreadedVelocityVerlet.hpp"

#include <cmath>

static float square(float val)
{
	return val * val;
}

void SingleThreadedVelocityVerlet::compute_forces()
{
	std::fill(m_new_forces.begin(), m_new_forces.end(), sf::Vector3f(0.f, 0.f, 0.f));

	for (size_t me = 0; me < m_num_particles - 1; ++me)
	{
		for (size_t other = me + 1; other < m_num_particles; ++other)
		{
			float sqr_distance = square(m_positions[other].x - m_positions[me].x);
			sqr_distance += square(m_positions[other].y - m_positions[me].y);
			sqr_distance += square(m_positions[other].z - m_positions[me].z);

			float gravity = m_masses[me] * m_masses[other] / (std::sqrt(sqr_distance) * sqr_distance);

			sf::Vector3f force;
			force.x = gravity * (m_positions[other].x - m_positions[me].x);
			force.y = gravity * (m_positions[other].y - m_positions[me].y);
			force.z = gravity * (m_positions[other].z - m_positions[me].z);

			m_new_forces[me] += force;

			m_new_forces[other] -= force;
		}
	}
}

void SingleThreadedVelocityVerlet::update_positions()
{
	for (size_t i = 0; i < m_num_particles; ++i)
	{
		float acceleration = m_time_step * 0.5f / m_masses[i];

		sf::Vector3f move;

		move.x = m_time_step * (m_velocities[i].x + acceleration * m_new_forces[i].x);
		move.y = m_time_step * (m_velocities[i].y + acceleration * m_new_forces[i].y);
		move.z = m_time_step * (m_velocities[i].z + acceleration * m_new_forces[i].z);

		m_positions[i] += move;

		m_old_forces[i] = m_new_forces[i];
	}
}

void SingleThreadedVelocityVerlet::update_velocities()
{
	for (size_t i = 0; i < m_num_particles; ++i)
	{
		float acceleration = m_time_step * 0.5f / m_masses[i];

		sf::Vector3f delta;
		
		delta.x = acceleration * (m_new_forces[i].x + m_old_forces[i].x);
		delta.y = acceleration * (m_new_forces[i].y + m_old_forces[i].y);
		delta.z = acceleration * (m_new_forces[i].z + m_old_forces[i].z);

		m_velocities[i] += delta;
	}
}

void SingleThreadedVelocityVerlet::initialize()
{
}

std::vector<sf::Vertex> SingleThreadedVelocityVerlet::run()
{
	compute_forces();
	update_positions();
	compute_forces();
	update_velocities();

	std::vector<sf::Vertex> vertices(m_num_particles);

	for (size_t i = 0; i < m_num_particles; ++i)
	{
		vertices[i] = sf::Vertex(sf::Vector2f(m_positions[i].x, m_positions[i].y));
	}

	return vertices;
}
