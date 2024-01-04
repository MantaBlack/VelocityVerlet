#ifndef SINGLE_GPU_VELOCITY_VERLET_HPP_
#define SINGLE_GPU_VELOCITY_VERLET_HPP_

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 220

#include "IAlgorithmStrategy.hpp"

#include <array>
#include <CL/opencl.hpp>
#include <SFML/System/Vector3.hpp>
#include <string>
#include <vector>

class SingleGPUVelocityVerlet : public IAlgorithmStrategy
{
private:
	const std::string KERNEL_FILE_NAME = "velocity_verlet.cl";
	const std::string FORCE_KERNEL_NAME = "compute_forces";
	const std::string VELOCITY_KERNEL_NAME = "compute_velocities";
	const std::string POSITIONS_KERNEL_NAME = "compute_positions";

	cl::Platform m_platform;
	std::string m_platform_name;
	std::string m_platform_vendor;
	cl::Context m_context;
	cl::Device m_device;
	std::string m_device_name;

	cl::Program::Sources m_source;
	cl::Program m_program;
	std::string m_build_options;
	std::string m_source_file;

	cl::CommandQueue m_command_queue;

	std::array<cl::Buffer, 2u> m_positions_buffers;
	std::array<cl::Buffer, 2u> m_forces_buffers;
	cl::Buffer m_velocities_buffer;

	cl::Kernel m_force_kernel;
	cl::Kernel m_velocities_kernel;
	cl::Kernel m_positions_kernel;

	std::vector<sf::Vector3f>& m_positions;
	std::vector<sf::Vector3f>& m_velocities;
	std::vector<float>& m_masses;
	float m_time_step;
	std::size_t m_num_particles;

	void validate_inputs();
	void setup_platform();
	void setup_context();
	void setup_device();
	void setup_program();
	void setup_command_queue();
	void setup_buffers();
	void setup_kernels();
	void queue_commands();
	void update_kernel_arguments();

public:
	SingleGPUVelocityVerlet(std::size_t num_particles,
		float time_step,
		std::vector<sf::Vector3f>& positions,
		std::vector<sf::Vector3f>& velocities,
		std::vector<float>& masses)
		: m_num_particles(num_particles),
		m_time_step(time_step),
		m_positions(positions),
		m_velocities(velocities),
		m_masses(masses)
	{}

	~SingleGPUVelocityVerlet()
	{}

	void initialize() override;
	std::vector<sf::Vertex> run() override;
};

#endif // !SINGLE_GPU_VELOCITY_VERLET_HPP_
