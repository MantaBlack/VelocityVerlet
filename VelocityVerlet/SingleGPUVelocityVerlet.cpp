#include "SingleGPUVelocityVerlet.hpp"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

bool SingleGPUVelocityVerlet::validate_inputs() const
{
	const std::vector<std::size_t> input_sizes =
	{
		m_input_positions.size(),
	    m_input_velocities.size(),
	    m_input_masses.size(),
	    m_num_particles
	};

	if (!std::equal(input_sizes.begin() + 1, input_sizes.end(), input_sizes.begin()))
	{
		return false;
	}

	return (m_num_particles > 0);
}

bool SingleGPUVelocityVerlet::setup_platform()
{
	try
	{
		std::vector<cl::Platform> platforms;

		cl::Platform::get(&platforms);

		if (platforms.empty())
		{
			return false;
		}

		m_platform = platforms.front();
		m_platform_name = m_platform.getInfo<CL_PLATFORM_NAME>();
		m_platform_vendor = m_platform.getInfo<CL_PLATFORM_VENDOR>();

		return true;
	}
	catch (const cl::Error& e)
	{
		std::cout << "Error: " << e.err() << std::endl;
		std::cout << "Exception: " << e.what() << std::endl;

		return false;
	}
}

bool SingleGPUVelocityVerlet::setup_context()
{
	try
	{
		cl_context_properties props[3] =
		{
			CL_CONTEXT_PLATFORM,
			(cl_context_properties)(m_platform)(),
			0
		};

		m_context = cl::Context(CL_DEVICE_TYPE_GPU, props, NULL, NULL);

		return true;
	}
	catch (const cl::Error& e)
	{
		std::cout << "Error: " << e.err() << std::endl;
		std::cout << "Exception: " << e.what() << std::endl;

		return false;
	}
}

bool SingleGPUVelocityVerlet::setup_device()
{
	try
	{
		std::vector<cl::Device> devices = m_context.getInfo<CL_CONTEXT_DEVICES>();

		if (devices.empty())
		{
			return false;
		}

		m_device = devices.front();

#ifdef CL_DEVICE_BOARD_NAME_AMD
		m_device_name = m_device.getInfo<CL_DEVICE_BOARD_NAME_AMD>();
#else
		m_device_name = m_device.getInfo<CL_DEVICE_NAME>();
#endif

		return true;
	}
	catch (const cl::Error& e)
	{
		std::cout << "Error: " << e.err() << std::endl;
		std::cout << "Exception: " << e.what() << std::endl;

		return false;
	}
}

bool SingleGPUVelocityVerlet::setup_program()
{
	try
	{
		std::ifstream file_stream(KERNEL_FILE_NAME);
		std::stringstream buffer;
		buffer << file_stream.rdbuf();
		
		std::string kernel_code(buffer.str());

		cl::Program::Sources sources{ kernel_code };

		m_program = cl::Program(m_context, kernel_code);
		m_program.build(m_device, BUILD_OPTIONS.data());

		return true;
	}
	catch (const cl::Error& e)
	{
		std::cout << "Error: " << e.err() << std::endl;
		std::cout << "Exception: " << e.what() << std::endl;

		std::cout << "Build status: " << m_program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(m_device) << std::endl;
		std::cout << "Build options: " << m_program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(m_device) << std::endl;
		std::cout << "Build log: " << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device) << std::endl;

		return false;
	}
	catch (const std::exception& e)
	{
		std::cout << "Exception: " << e.what() << std::endl;

		return false;
	}
}

bool SingleGPUVelocityVerlet::setup_command_queue()
{
	try
	{
		m_command_queue = cl::CommandQueue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE, NULL);

		return true;
	}
	catch (const cl::Error& e)
	{
		std::cout << "Error: " << e.err() << std::endl;
		std::cout << "Exception: " << e.what() << std::endl;

		return false;
	}
}

bool SingleGPUVelocityVerlet::setup_input_data()
{
	m_buffer_size_bytes = m_num_particles * sizeof(cl_float4);

	m_positions = (cl_float4*)_aligned_malloc(m_buffer_size_bytes, 16);
	m_velocities = (cl_float4*)_aligned_malloc(m_buffer_size_bytes, 16);

	if ((m_positions == nullptr) || (m_velocities == nullptr))
	{
		return false;
	}

	const std::size_t dims = 4;

	for (std::size_t i = 0; i < m_num_particles; ++i)
	{
		// copy position values
		m_positions[i].s0 = m_input_positions[i].x;
		m_positions[i].s1 = m_input_positions[i].y;
		m_positions[i].s2 = m_input_positions[i].z;
		
		// the 4th component contains the mass for this particle
		m_positions[i].s3 = m_input_masses[i];

		// copy velocity values
		m_velocities[i].s0 = m_input_velocities[i].x;
		m_velocities[i].s1 = m_input_velocities[i].y;
		m_velocities[i].s2 = m_input_velocities[i].z;

		// the 4th component also contains the mass the this particle
		m_velocities[i].s3 = m_input_masses[i];
	}

	return true;
}

bool SingleGPUVelocityVerlet::setup_buffers()
{
	try
	{
		// create buffers for storing positions and masses
		m_positions_buffers[m_front_buffer_idx] = cl::Buffer(m_context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			m_buffer_size_bytes,
			m_positions);

		m_positions_buffers[m_back_buffer_idx] = cl::Buffer(m_context,
			CL_MEM_READ_WRITE,
			m_buffer_size_bytes,
			NULL);

		// create buffer for storing velocity values
		m_velocities_buffer = cl::Buffer(m_context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			m_buffer_size_bytes,
			m_velocities);

		// output buffer for copying results from device (this should be a pinned buffer)
		m_output_buffer = cl::Buffer(m_context, CL_MEM_USE_HOST_PTR, m_buffer_size_bytes, m_positions);

		return true;
	}
	catch (const cl::Error& e)
	{
		std::cout << "Error: " << e.err() << std::endl;
		std::cout << "Exception: " << e.what() << std::endl;

		return false;
	}
}

bool SingleGPUVelocityVerlet::setup_kernels()
{
	try
	{
		m_force_kernel = cl::Kernel(m_program, FORCE_KERNEL_NAME.data());

		m_force_kernel.setArg(0, m_forces_buffers[m_front_buffer_idx]);
		m_force_kernel.setArg(1, m_positions_buffers[m_front_buffer_idx]);
		m_force_kernel.setArg(2, WORKGROUP_SIZE * sizeof(cl_float4), NULL);

		m_positions_kernel = cl::Kernel(m_program, POSITIONS_KERNEL_NAME.data());

		m_positions_kernel.setArg(0, m_forces_buffers[m_front_buffer_idx]);
		m_positions_kernel.setArg(1, m_positions_buffers[m_front_buffer_idx]);
		m_positions_kernel.setArg(2, m_positions_buffers[m_back_buffer_idx]);
		m_positions_kernel.setArg(3, m_velocities_buffer);
		m_positions_kernel.setArg(4, m_time_step);

		m_velocities_kernel = cl::Kernel(m_program, VELOCITY_KERNEL_NAME.data());

		m_velocities_kernel.setArg(0, m_forces_buffers[m_front_buffer_idx]);
		m_velocities_kernel.setArg(1, m_forces_buffers[m_back_buffer_idx]);
		m_velocities_kernel.setArg(2, m_velocities_buffer);
		m_velocities_kernel.setArg(3, m_time_step);

		return true;
	}
	catch (const cl::Error& e)
	{
		std::cout << "Error: " << e.err() << std::endl;
		std::cout << "Exception: " << e.what() << std::endl;

		return false;
	}
}

bool SingleGPUVelocityVerlet::update_kernel_arguments()
{
	try
	{
		std::swap(m_front_buffer_idx, m_back_buffer_idx);

		m_force_kernel.setArg(0, m_forces_buffers[m_front_buffer_idx]);
		m_force_kernel.setArg(1, m_positions_buffers[m_front_buffer_idx]);

		m_positions_kernel.setArg(0, m_forces_buffers[m_front_buffer_idx]);
		m_positions_kernel.setArg(1, m_positions_buffers[m_front_buffer_idx]);
		m_positions_kernel.setArg(2, m_positions_buffers[m_back_buffer_idx]);

		return true;
	}
	catch (const cl::Error& e)
	{
		std::cout << "Error: " << e.err() << std::endl;
		std::cout << "Exception: " << e.what() << std::endl;

		return false;
	}
}

bool SingleGPUVelocityVerlet::queue_commands()
{
	try
	{
		std::vector<cl::Event> pos_event(1);
		std::vector<cl::Event> force_event(1);
		std::vector<cl::Event> vel_event(1);
		std::vector<cl::Event> read_event(1);

		if (m_command_queue.has_value())
		{
			// computes forces
			m_command_queue->enqueueNDRangeKernel(m_force_kernel,
				cl::NullRange,
				cl::NDRange(m_total_workitems),
				cl::NDRange(WORKGROUP_SIZE),
				NULL,
				&force_event[0]);

			// compute positions
			m_command_queue->enqueueNDRangeKernel(m_positions_kernel,
				cl::NullRange,
				cl::NDRange(m_total_workitems),
				cl::NDRange(WORKGROUP_SIZE),
				&force_event,
				&pos_event[0]);

			// we don't wanna override the newly computed forces so we switch buffers
			m_force_kernel.setArg(0, m_forces_buffers[m_back_buffer_idx]);

			// set the buffer that contains the newly updated positions
			m_force_kernel.setArg(1, m_positions_buffers[m_back_buffer_idx]);

			// re-compute forces after updating positions
			m_command_queue->enqueueNDRangeKernel(m_force_kernel,
				cl::NullRange,
				cl::NDRange(m_total_workitems),
				cl::NDRange(WORKGROUP_SIZE),
				&pos_event,
				&force_event[0]);

			// compute velocities
			m_command_queue->enqueueNDRangeKernel(m_velocities_kernel,
				cl::NullRange,
				cl::NDRange(m_total_workitems),
				cl::NDRange(WORKGROUP_SIZE),
				&force_event,
				&vel_event[0]);

			// copy positions from device
			m_command_queue->enqueueCopyBuffer(m_positions_buffers[m_back_buffer_idx],
				m_output_buffer,
				0, // source offset
				0, // destination offset
				m_buffer_size_bytes,
				&force_event,
				&read_event[0]);

			// sync
			m_command_queue->finish();

			// swap buffers and update kernel arguments
			if (!update_kernel_arguments())
			{
				return false;
			}
		}

		return true;
	}
	catch (const cl::Error& e)
	{
		std::cout << "Error: " << e.err() << std::endl;
		std::cout << "Exception: " << e.what() << std::endl;

		return false;
	}
}

void SingleGPUVelocityVerlet::initialize()
{
	if (!validate_inputs())
	{
		throw std::string("Failure due to invalid inputs");
	}

	if (setup_platform())
	{
		std::cout << "Platform setup is OK" << std::endl;
		std::cout << "Platform name   : " << m_platform_name << std::endl;
		std::cout << "Platform vendor : " << m_platform_vendor << std::endl;
	}
	else
	{
		throw std::string("Failed to setup platform");
	}

	if (!setup_context())
	{
		throw std::string("Failed to setup context");
	}

	if (setup_device())
	{
		std::cout << "Device setup is OK" << std::endl;
		std::cout << "Device name : " << m_device_name.value() << std::endl;
	}
	else
	{
		throw std::string("Failed to setup context");
	}
}

std::vector<sf::Vertex> SingleGPUVelocityVerlet::run()
{
	return std::vector<sf::Vertex>();
}
