#ifndef SINGLE_GPU_VELOCITY_VERLET_HPP_
#define SINGLE_GPU_VELOCITY_VERLET_HPP_

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 220

#include "IAlgorithmStrategy.hpp"

#include <array>
#include <CL/opencl.hpp>
#include <cstdlib>
#include <optional>
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
	const std::string BUILD_OPTIONS = "-cl-std=CL2.2";
	const std::size_t WORKGROUP_SIZE = 256u;

	cl::Platform m_platform;
	std::string m_platform_name;
	std::string m_platform_vendor;
	cl::Context m_context;
	cl::Device m_device;
	std::optional<std::string> m_device_name;

	cl::Program::Sources m_source;
	cl::Program m_program;
	std::string m_source_file;

	std::optional<cl::CommandQueue> m_command_queue;

	std::array<cl::Buffer, 2u> m_positions_buffers;
	std::array<cl::Buffer, 2u> m_forces_buffers;
	cl::Buffer m_velocities_buffer;
	cl::Buffer m_output_buffer;
	std::size_t m_buffer_size_bytes;
	std::size_t m_front_buffer_idx;
	std::size_t m_back_buffer_idx;

	cl::Kernel m_force_kernel;
	cl::Kernel m_velocities_kernel;
	cl::Kernel m_positions_kernel;

	std::vector<sf::Vector3f>& m_input_positions;
	std::vector<sf::Vector3f>& m_input_velocities;
	std::vector<float>& m_input_masses;

	cl_float4* m_positions;
	cl_float4* m_velocities;

	float m_time_step;
	std::size_t m_num_particles;

	std::size_t m_total_workitems;

	bool validate_inputs() const;
	bool setup_platform();
	bool setup_context();
	bool setup_device();
	bool setup_program();
	bool setup_command_queue();
	bool setup_input_data();
	bool setup_buffers();
	bool setup_kernels();
	bool update_kernel_arguments();
	bool queue_commands();

public:
	SingleGPUVelocityVerlet(std::size_t num_particles,
		float time_step,
		std::vector<sf::Vector3f>& positions,
		std::vector<sf::Vector3f>& velocities,
		std::vector<float>& masses)
		: m_num_particles(num_particles),
		m_time_step(time_step),
		m_input_positions(positions),
		m_input_velocities(velocities),
		m_input_masses(masses),
		m_positions(nullptr),
		m_velocities(nullptr),
		m_front_buffer_idx(0u),
		m_back_buffer_idx(1u),
		m_buffer_size_bytes(0u),
		m_total_workitems(num_particles)
	{}

	~SingleGPUVelocityVerlet()
	{
		if (m_command_queue.has_value())
		{
			m_command_queue->finish();
		}

		if (m_positions != nullptr)
		{
			_aligned_free(m_positions);
			m_positions = nullptr;
		}

		if (m_velocities != nullptr)
		{
			_aligned_free(m_velocities);
			m_velocities = nullptr;
		}
	}

	void initialize() override;
	std::vector<sf::Vertex> run() override;
};

#endif // !SINGLE_GPU_VELOCITY_VERLET_HPP_
