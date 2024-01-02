#include "VertexBufferRenderer.hpp"

#include <string>

void VertexBufferRenderer::draw(sf::RenderTarget& target, sf::RenderStates states) const
{
	states.transform *= getTransform();

	states.texture = NULL;

	target.draw(m_vertex_buffer, states);
}

void VertexBufferRenderer::update(std::vector<sf::Vertex> vertices)
{
	if (m_vertex_buffer.getVertexCount() == vertices.size())
	{
		m_vertex_buffer.update(&vertices.front());
	}
	else
	{
		if (!m_vertex_buffer.create(vertices.size()))
		{
			throw "Failed to create vertex buffer. Vertex count: " + std::to_string(vertices.size());
		}

		if (!m_vertex_buffer.update(&vertices.front()))
		{
			throw "Failed to update vertex buffer";
		}
	}
}
