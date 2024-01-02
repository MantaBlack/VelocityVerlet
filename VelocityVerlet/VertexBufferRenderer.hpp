#ifndef VERTEX_BUFFER_RENDERER_HPP_
#define VERTEX_BUFFER_RENDERER_HPP_

#include <SFML/Graphics.hpp>
#include <vector>

class VertexBufferRenderer : public sf::Drawable, public sf::Transformable
{
private:
	std::vector<sf::Vertex> m_vertices;
	sf::VertexBuffer::Usage m_usage;
	sf::PrimitiveType m_primitive_type;
	sf::VertexBuffer m_vertex_buffer;

	void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

public:
	VertexBufferRenderer(std::vector<sf::Vertex> vertices,
		sf::VertexBuffer::Usage usage,
		sf::PrimitiveType primitive_type)
		: m_vertices(vertices),
		m_usage(usage),
		m_primitive_type(primitive_type),
		m_vertex_buffer(primitive_type, usage)
	{ }

	~VertexBufferRenderer()
	{
		m_vertex_buffer.create(0);
	}

	void update(std::vector<sf::Vertex> vertices);
};

#endif // !VERTEX_BUFFER_RENDERER_HPP_
