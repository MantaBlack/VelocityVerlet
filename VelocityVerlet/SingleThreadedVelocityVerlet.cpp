#include "SingleThreadedVelocityVerlet.hpp"

#include <cmath>

static float square(float val)
{
	return val * val;
}

void STVV::SingleThreadedVelocityVerlet::compute_forces()
{
	std::fill(m_new_forces.begin(), m_new_forces.end(), sf::Vector3f(0.f, 0.f, 0.f));

	for (size_t me = 0; me < m_num_particles; ++me)
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
