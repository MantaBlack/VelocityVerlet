#ifndef VERTEX_BUFFER_RENDERER_HPP_
#define VERTEX_BUFFER_RENDERER_HPP_

#include "IRenderStrategy.hpp"

#include <SFML/Graphics.hpp>
#include <vector>

class VertexBufferRenderer : public IRenderStrategy
{
private:
	sf::VertexBuffer::Usage m_usage;
	sf::PrimitiveType m_primitive_type;
	sf::VertexBuffer m_vertex_buffer;

public:
	VertexBufferRenderer(sf::VertexBuffer::Usage usage,
		sf::PrimitiveType primitive_type)
		: m_usage(usage),
		m_primitive_type(primitive_type),
		m_vertex_buffer(primitive_type, usage)
	{ }

	~VertexBufferRenderer()
	{
		m_vertex_buffer.create(0);
	}

	void update(std::vector<sf::Vertex> vertices) override;

	const sf::Drawable& get_frame() const override;
};

#endif // !VERTEX_BUFFER_RENDERER_HPP_
