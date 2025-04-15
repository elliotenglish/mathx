#ifndef RENDERER_HPP
#define RENDERER_HPP

#define GL_GLEXT_PROTOTYPES

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#if defined(__APPLE__)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <unistd.h>

//https://github.com/capnramses/antons_opengl_tutorials_book/blob/master/00_hello_triangle/main.c
//https://people.eecs.ku.edu/~jrmiller/Courses/672/InClass/3DModeling/glDrawElements.html


class Renderer
{
private:
// Shader sources
  static const GLchar* vertex_position_source;
  // static const GLchar* point_line_color_source;
  static const GLchar* triangle_color_source;
  
  GLuint vertex_position_shader;
  // GLuint point_line_color_shader;
  GLuint triangle_color_shader;
  GLuint shader_program;

  GLuint position_attr;
  GLuint color_attr;
  GLuint normal_attr;
  GLuint camera_attr;
  GLuint projection_attr;

  GLuint vertex_array;
  GLuint vertex_position_buffer;
  GLuint vertex_color_buffer;
  GLuint vertex_normal_buffer;
  GLuint element_index_buffer;

  Eigen::Vector2i window_dimensions;
  SDL_Window* window;
  SDL_GLContext context;

  Eigen::Affine3f camera_transform;
  Eigen::Matrix4f camera_matrix;
  Eigen::Matrix4f projection_matrix;

public:
  Renderer();
  ~Renderer();

  void DrawPoint(const Eigen::Vector3f& x0,const float radius,
                 const Eigen::Vector3f& c0);
  void DrawLine(const Eigen::Vector3f& x0,const Eigen::Vector3f& x1,const float radius,
                const Eigen::Vector3f& c0,const Eigen::Vector3f& c1);
  void DrawTriangle(const Eigen::Vector3f& x0,const Eigen::Vector3f& x1,const Eigen::Vector3f& x2,
                    const Eigen::Vector3f& c0,const Eigen::Vector3f& c1,const Eigen::Vector3f& c2);
//radius - line thickness, c- color, the function calls line
  //void DrawBox(const Box& b,const float radius,const Eigen::Vector3f& c);

  void ClearBuffer(const Eigen::Vector3f& color);

  /** @return Time in seconds. This is only accurate to milliseconds (1e-3). We should have at least microsecond units. */
  double GetTime();
  bool PollEvent(SDL_Event& event);
  void SwapBuffers();
  void SetFullscreen(const bool fullscreen);

  Eigen::Affine3f& CameraTransform();
};

#endif
