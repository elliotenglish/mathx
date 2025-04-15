#define GL_GLEXT_PROTOTYPES

//#include <OpenGL/gl3.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <unistd.h>
#include "Renderer.hpp"
#include "utilities/Source/Logger.hpp"

/**
 * https://open.gl/content/code/c2_triangle_elements.txt
 */

void CheckCompileShader(GLuint shader)
{
  int  success;
  char infoLog[512];
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if(success==GL_FALSE)
  {
    glGetShaderInfoLog(shader, 512, NULL, infoLog);
    std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    assert(0);
  }
}

void CheckLinkProgram(GLuint program)
{
  int success;
  char infoLog[512];
  glGetProgramiv(program,GL_LINK_STATUS,&success);
  if(success==GL_FALSE)
  {
    glGetProgramInfoLog(program, 512, NULL, infoLog);
    std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    assert(0);
  }
}

GLuint CompileShader(const GLchar* source,const GLuint type)
{
  GLuint shader=glCreateShader(type);
  glShaderSource(shader,1,&source,NULL);
  glCompileShader(shader);
  CheckCompileShader(shader);
  return shader;
}

Renderer::Renderer()
{
  SDL_Init(SDL_INIT_EVERYTHING);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

  window_dimensions=Eigen::Vector2i(800,800);
  window=SDL_CreateWindow("sim",0,0,window_dimensions(0),window_dimensions(1),SDL_WINDOW_OPENGL);
  assert(window);
  context=SDL_GL_CreateContext(window);
  SDL_GL_SetSwapInterval(0);//Needed to stop SDL from using vsync and slowing down the swap window.

  std::cout << "OpenGL version=" << glGetString(GL_VERSION) << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Create and compile the position shader
  vertex_position_shader=CompileShader(vertex_position_source,GL_VERTEX_SHADER);
  triangle_color_shader=CompileShader(triangle_color_source,GL_FRAGMENT_SHADER);

  // Link the vertex and fragment shader into a shader program
  shader_program = glCreateProgram();
  glAttachShader(shader_program, vertex_position_shader);
  glAttachShader(shader_program, triangle_color_shader);
  glLinkProgram(shader_program);
  CheckLinkProgram(shader_program);

  //////////////////////////////////////////////////////////////////////////////
  //Get attribute indexes
  position_attr=glGetAttribLocation(shader_program,"position");
  color_attr=glGetAttribLocation(shader_program,"color");
  normal_attr=glGetAttribLocation(shader_program,"normal");
  camera_attr=glGetUniformLocation(shader_program,"camera");
  projection_attr=glGetUniformLocation(shader_program,"projection");

  std::cout << "attributes" << " position=" << position_attr << " color=" << color_attr << " normal=" << normal_attr << " camera=" << camera_attr << " projection=" << projection_attr << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  //Generate data buffers
  glGenVertexArrays(1,&vertex_array);
  glGenBuffers(1,&vertex_position_buffer);
  glGenBuffers(1,&vertex_color_buffer);
  glGenBuffers(1,&vertex_normal_buffer);
  glGenBuffers(1,&element_index_buffer);

  //////////////////////////////////////////////////////////////////////////////
  //Initialize camera
  camera_transform=Eigen::Affine3f::Identity();
}

Renderer::~Renderer()
{
  glDeleteProgram(shader_program);
  glDeleteShader(triangle_color_shader);
  glDeleteShader(vertex_position_shader);

  SDL_GL_DeleteContext(context);
  SDL_DestroyWindow(window);
  SDL_Quit();
}

void Renderer::DrawPoint(const Eigen::Vector3f& x0,const float radius,
                         const Eigen::Vector3f& c0)
{
  //https://en.wikipedia.org/wiki/Tetrahedron#Coordinates_for_a_regular_tetrahedron
  Eigen::Vector3f v0=x0+radius*Eigen::Vector3f(sqrt(8./9.),0,-1./3.);
  Eigen::Vector3f v1=x0+radius*Eigen::Vector3f(-sqrt(2./9.),sqrt(2./3.),-1./3.);
  Eigen::Vector3f v2=x0+radius*Eigen::Vector3f(-sqrt(2./9.),-sqrt(2./3.),-1./3.);
  Eigen::Vector3f v3=x0+radius*Eigen::Vector3f(0,0,1);

  DrawTriangle(v0,v1,v2,c0,c0,c0);
  DrawTriangle(v0,v2,v3,c0,c0,c0);
  DrawTriangle(v0,v3,v1,c0,c0,c0);
  DrawTriangle(v1,v3,v2,c0,c0,c0);
}

void Renderer::DrawLine(const Eigen::Vector3f& x0,const Eigen::Vector3f& x1,const float radius,
                        const Eigen::Vector3f& c0,const Eigen::Vector3f& c1)
{
  Eigen::Vector3f dir=x1-x0;
  dir/=dir.norm();

  int min_axis=0;
  for(int d=1;d<3;d++)
    if(fabs(dir(d))<fabs(dir(min_axis)))
      min_axis=d;
  Eigen::Vector3f ortho0(0,0,0);
  ortho0(min_axis)=1;
  ortho0-=dir*dir.transpose()*ortho0;
  ortho0/=ortho0.norm();

  Eigen::Vector3f ortho1=ortho0.cross(dir);
  ortho1/=ortho1.norm();

  Eigen::Vector3f o0=ortho0;
  Eigen::Vector3f o1=cos(M_PI*2./3.)*ortho0+sin(M_PI*2./3.)*ortho1;
  Eigen::Vector3f o2=cos(M_PI*4./3.)*ortho0+sin(M_PI*4./3.)*ortho1;

  Eigen::Vector3f v0=x0+radius*o0;
  Eigen::Vector3f v1=x0+radius*o1;
  Eigen::Vector3f v2=x0+radius*o2;
  Eigen::Vector3f v3=x1+radius*o0;
  Eigen::Vector3f v4=x1+radius*o1;
  Eigen::Vector3f v5=x1+radius*o2;

  //Side 0
  DrawTriangle(v0,v1,v3,c0,c0,c1);
  DrawTriangle(v1,v4,v3,c0,c1,c1);

  //Side 1
  DrawTriangle(v1,v2,v4,c0,c0,c1);
  DrawTriangle(v2,v5,v4,c0,c1,c1);

  //Side 2
  DrawTriangle(v2,v0,v5,c0,c0,c1);
  DrawTriangle(v0,v3,v5,c0,c1,c1);

  //End 0
  DrawTriangle(v0,v1,v2,c0,c0,c0);

  //End 1
  DrawTriangle(v3,v5,v4,c1,c1,c1);
}

// void Renderer::DrawBox(const Box& b,const float radius,const Eigen::Vector3f& c)
// {
//   Renderer::DrawLine((Eigen::Vector3f(b.min(0),b.min(1),b.min(2))),(Eigen::Vector3f(b.max(0),b.min(1),b.min(2))),radius,c,c);
//   Renderer::DrawLine((Eigen::Vector3f(b.min(0),b.min(1),b.min(2))),(Eigen::Vector3f(b.min(0),b.max(1),b.min(2))),radius,c,c);
//   Renderer::DrawLine((Eigen::Vector3f(b.max(0),b.min(1),b.min(2))),(Eigen::Vector3f(b.max(0),b.max(1),b.min(2))),radius,c,c);
//   Renderer::DrawLine((Eigen::Vector3f(b.min(0),b.max(1),b.min(2))),(Eigen::Vector3f(b.max(0),b.max(1),b.min(2))),radius,c,c);

//   Renderer::DrawLine((Eigen::Vector3f(b.min(0),b.min(1),b.max(2))),(Eigen::Vector3f(b.max(0),b.min(1),b.max(2))),radius,c,c);  
//   Renderer::DrawLine((Eigen::Vector3f(b.min(0),b.min(1),b.max(2))),(Eigen::Vector3f(b.min(0),b.max(1),b.max(2))),radius,c,c);
//   Renderer::DrawLine((Eigen::Vector3f(b.max(0),b.min(1),b.max(2))),(Eigen::Vector3f(b.max(0),b.max(1),b.max(2))),radius,c,c);
//   Renderer::DrawLine((Eigen::Vector3f(b.min(0),b.max(1),b.max(2))),(Eigen::Vector3f(b.max(0),b.max(1),b.max(2))),radius,c,c);

//   Renderer::DrawLine((Eigen::Vector3f(b.min(0),b.min(1),b.min(2))),(Eigen::Vector3f(b.min(0),b.min(1),b.max(2))),radius,c,c);  
//   Renderer::DrawLine((Eigen::Vector3f(b.max(0),b.min(1),b.min(2))),(Eigen::Vector3f(b.max(0),b.min(1),b.max(2))),radius,c,c);
//   Renderer::DrawLine((Eigen::Vector3f(b.min(0),b.max(1),b.min(2))),(Eigen::Vector3f(b.min(0),b.max(1),b.max(2))),radius,c,c);
//   Renderer::DrawLine((Eigen::Vector3f(b.max(0),b.max(1),b.min(2))),(Eigen::Vector3f(b.max(0),b.max(1),b.max(2))),radius,c,c);
// }

void Renderer::DrawTriangle(const Eigen::Vector3f& x0,const Eigen::Vector3f& x1,const Eigen::Vector3f& x2,
                            const Eigen::Vector3f& c0,const Eigen::Vector3f& c1,const Eigen::Vector3f& c2)
{
  static int count=0;
  count++;
  // if(count%10000==0)
  //   Logger << "Drawing triangle x0=" << x0.transpose() << " x1=" << x1.transpose() << " x2=" << x2.transpose() << " count=" << count << std::endl;

  GLuint indexes[] = {
    0, 1, 2
  };

  GLfloat positions[] = {
    x0(0),x0(1),x0(2),
    x1(0),x1(1),x1(2),
    x2(0),x2(1),x2(2)
  };

  GLfloat colors[] = {
    c0(0),c0(1),c0(2),1,
    c1(0),c1(1),c1(2),1,
    c2(0),c2(1),c2(2),1
  };

  Eigen::Vector3f edge0=(x1-x0);
  Eigen::Vector3f edge1=(x2-x0);
  Eigen::Vector3f normal=edge0.cross(edge1);
  normal/=normal.norm();

  GLfloat normals[] = {
    normal(0),normal(1),normal(2),
    normal(0),normal(1),normal(2),
    normal(0),normal(1),normal(2)
  };

  //Column major
  // camera_matrix <<
  //   1,0,0,0,
  //   0,1,0,0,
  //   0,0,1,2,
  //   0,0,0,1;
  camera_matrix=camera_transform.matrix();

  projection_matrix <<
    std::min(1.f,(float)window_dimensions(1)/window_dimensions(0)),0,0,0,
    0,std::min(1.f,(float)window_dimensions(0)/window_dimensions(1)),0,0,
    0,0,0,0,
    0,0,1,0;

  // Logger << "camera_matrix=" << std::endl << camera_matrix << std::endl;
  // Logger << "projection_matrix=" << std::endl << projection_matrix << std::endl;

  //Setup arrays
  glBindVertexArray(vertex_array);

  //Setup the position buffer
  glEnableVertexAttribArray(position_attr);
  glBindBuffer(GL_ARRAY_BUFFER,vertex_position_buffer);
  glBufferData(GL_ARRAY_BUFFER,sizeof(positions),positions,GL_DYNAMIC_DRAW);
  glVertexAttribPointer(position_attr,3,GL_FLOAT,GL_FALSE,0,NULL);

  //Set the color buffer
  glEnableVertexAttribArray(color_attr);
  glBindBuffer(GL_ARRAY_BUFFER,vertex_color_buffer);
  glBufferData(GL_ARRAY_BUFFER,sizeof(colors),colors,GL_DYNAMIC_DRAW);
  glVertexAttribPointer(color_attr,4,GL_FLOAT,GL_FALSE,0,NULL);

  //Set the normal buffer
  glEnableVertexAttribArray(normal_attr);
  glBindBuffer(GL_ARRAY_BUFFER,vertex_normal_buffer);
  glBufferData(GL_ARRAY_BUFFER,sizeof(normals),normals,GL_DYNAMIC_DRAW);
  glVertexAttribPointer(normal_attr,3,GL_FLOAT,GL_FALSE,0,NULL);

  //Configure the element indexes
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,element_index_buffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(indexes),indexes,GL_DYNAMIC_DRAW);

  //Configure camera matrix
  glUniformMatrix4fv(camera_attr,1,GL_FALSE,camera_matrix.data());

  //Configure camera matrix
  glUniformMatrix4fv(projection_attr,1,GL_FALSE,projection_matrix.data());

  //Enable program and draw
  glUseProgram(shader_program);
  glBindVertexArray(vertex_array);
  glDrawElements(GL_TRIANGLES,3,GL_UNSIGNED_INT,NULL);
  return;

  //glDrawArrays(GL_TRIANGLES,0,3);
}

// Shader sources
const GLchar* Renderer::vertex_position_source =
"#version 410\n"
"  uniform mat4 camera;\n"
"  uniform mat4 projection;\n"
"  in vec3 position;\n"
"  in vec3 color;\n"
"  in vec3 normal;\n"
"  out vec3 out_color;\n"
"  out vec3 out_normal;\n"
"  void main()\n"
"  {\n"
"    out_color=color;\n"
"    out_normal=normal;\n"
"    gl_Position=projection*camera*vec4(position,1);\n"
//"    gl_Position=vec4(position,1);\n"
"  }\n";

// const GLchar* Renderer::point_line_color_source =
// "#version 410\n"
// "  in vec3 out_color;\n"
// "  out vec4 frag_color;\n"
// "  void main()\n"
// "  {\n"
// //"    frag_color=vec4(1,1,1,1);\n"
// "    frag_color=vec4(out_color,1);\n"
// "  }\n";

//TODO: Make this a light-angle shader
const GLchar* Renderer::triangle_color_source =
"#version 410\n"
"  in vec3 out_color;\n"
"  out vec4 frag_color;\n"
"  void main()\n"
"  {\n"
//"    frag_color=vec4(1,1,1,1);\n"
"    frag_color=vec4(out_color,1);\n"
"  }\n";

bool Renderer::PollEvent(SDL_Event& event)
{
  return SDL_PollEvent(&event);
}

void Renderer::ClearBuffer(const Eigen::Vector3f& color)
{
  SDL_GetWindowSize(window,&window_dimensions(0),&window_dimensions(1));
  //int dim=std::min(window_dimensions(0),window_dimensions(1));
  //glViewport((window_dimensions(0)-dim)/2,(window_dimensions(1)-dim)/2,dim,dim);
  glViewport(0,0,window_dimensions(0),window_dimensions(1));
  glClearColor(color(0),color(1),color(2),0.f);
  glClear(GL_COLOR_BUFFER_BIT);
  //SDL_RenderClear(renderer);
}

void Renderer::SwapBuffers()
{
  SDL_GL_SwapWindow(window);
}

double Renderer::GetTime()
{
  uint32_t time_ms=SDL_GetTicks();
  return time_ms*1e-3;
}

Eigen::Affine3f& Renderer::CameraTransform()
{
  return camera_transform;
}

void Renderer::SetFullscreen(const bool fullscreen)
{
  SDL_SetWindowFullscreen(window,fullscreen?SDL_WINDOW_FULLSCREEN:0);
}
