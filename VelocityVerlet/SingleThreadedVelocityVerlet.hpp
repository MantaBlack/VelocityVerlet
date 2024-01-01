#ifndef SINGLE_THREADED_VELOCITY_VERLET_HPP_
#define SINGLE_THREADED_VELOCITY_VERLET_HPP_

#include "IAlgorithmStrategy.hpp"

#include <SFML/System/Vector3.hpp>
#include <vector>

namespace STVV
{
	class SingleThreadedVelocityVerlet : public IAlgorithmStrategy
	{
	public:
		SingleThreadedVelocityVerlet(std::size_t num_particles,
			float time_step,
			std::vector<sf::Vector3f> positions,
			std::vector<sf::Vector3f> velocities,
			std::vector<float> masses)
			: m_num_particles(num_particles),
			m_time_step(time_step),
			m_positions(positions),
			m_velocities(velocities),
			m_masses(masses),
			m_old_forces(m_num_particles, sf::Vector3f(0.f, 0.f, 0.f)),
			m_new_forces(m_num_particles, sf::Vector3f(0.f, 0.f, 0.f))
		{}

		std::vector<sf::Vector3f> run() const override;

	private:
		void compute_forces();
		void update_positions();
		void update_velocities();

		std::vector<sf::Vector3f> m_positions;
		std::vector<sf::Vector3f> m_velocities;
		std::vector<sf::Vector3f> m_old_forces;
		std::vector<sf::Vector3f> m_new_forces;
		std::vector<float>        m_masses;

		float m_time_step;

		std::size_t m_num_particles;
	};
}

#endif // !SINGLE_THREADED_VELOCITY_VERLET_HPP_
