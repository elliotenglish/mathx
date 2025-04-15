#define GL_GLEXT_PROTOTYPES

//#include <OpenGL/gl3.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <Eigen/Dense>
#include <unistd.h>
#include "Renderer.hpp"
#include <picojson.h>
#include "utilities/Source/Logger.hpp"
#include "utilities/Source/PicoJSONUtilities.hpp"
#include "utilities/Source/ArghUtilities.hpp"
#include "utilities/Source/Time.hpp"

using namespace Code;

typedef Eigen::Vector3f V3;

V3 Color(const float a,const float z)
{
  return V3(.5+.5*cos(a)*cos(z),
            .5+.5*sin(a+M_PI)*cos(z+M_PI),
            .5+.5*cos(2*a)*cos(2*z));
}

V3 Position(const float a,const float z,const float radius)
{
  return V3(radius*cos(a)+.25*sin(z),
            radius*sin(a)+.25*cos(z*1.3211),
            z);
}

int main(int argc,const char** argv)
{
  //uint64_t start_time=stm_now();

  //////////////////////////////////////////////////////////////////////////////
  //https://github.com/hbristow/argparse
  picojson::value params=ArghToPicoJSON(argh::parser(argv));

  bool visualize_fullscreen=PicoJSONToBoolean(params,"visualize_fullscreen",false);
  float speed=PicoJSONToScalar(params,"speed",1);

  //Visualisation
  std::shared_ptr<Renderer> renderer;
  renderer.reset(new Renderer);
  renderer->SetFullscreen(visualize_fullscreen);
  renderer->CameraTransform().translate(Eigen::Vector3f(0,0,0));

  //  float dt=1e-3;  //add dependance on v
  int iteration=0;
  Time time=GetTime();
  float z_offset=0;

  bool running=true;

  while(running)
  {
    iteration++;
    
    Time new_time=GetTime();
    Time dt=new_time-time;
    time=new_time;

    //LogVar(new_time);
    //LogVar(dt);

    SDL_Event event;
    while(renderer->PollEvent(event))
    {
      //Log("event type=%d sym=%d %d %d",event.type,event.key.keysym.sym,SDL_KEYUP,SDLK_w);
      switch(event.type)
      {
      case SDL_KEYUP:
        if(!event.key.keysym.mod && event.key.keysym.sym==SDLK_f)
        {
          visualize_fullscreen=!visualize_fullscreen;
          renderer->SetFullscreen(visualize_fullscreen);
        }
        else if(event.key.keysym.sym==SDLK_w)
        {
          speed*=1.1;
        }
        else if(event.key.keysym.sym==SDLK_s)
        {
          speed*=1./1.1;
        }
        break;
      case SDL_QUIT:
        running=false;
        break;
      // case SDL_MOUSEMOTION:
      //   if(event.motion.state==SDL_BUTTON_LMASK)
      //   {
      //     Eigen::Affine3f& trans=renderer->CameraTransform();
      //     if(SDL_GetModState()&KMOD_CTRL)
      //     {
      //       trans.pretranslate(Eigen::Vector3f(event.motion.xrel*.01,0,event.motion.yrel*-.01));
      //     }
      //     else
      //     {
      //       trans.prerotate(Eigen::AngleAxis<float>(event.motion.xrel*.001,Eigen::Vector3f(0,1,0)));
      //       trans.prerotate(Eigen::AngleAxis<float>(event.motion.yrel*.001,Eigen::Vector3f(1,0,0)));
      //     }
      //   }
      //   break;
      default:
        break;
      }
    }

    renderer->ClearBuffer(V3(0,0,0));

    float z_size=20;
    float radius=1;
    int z_segs=64;
    int a_segs=64;
    z_offset+=speed*4*dt;
    //LogVar(speed);
    for(int z=0;z<z_segs;z++)
      for(int a=0;a<a_segs;a++)
      {
        float a0=(float)a/a_segs*2*M_PI;
        float a1=(float)(a+1)/a_segs*2*M_PI;
        float z0=(float)z/z_segs*z_size;
        float z1=float(z+1)/z_segs*z_size;
        V3 x00=Position(a0,z0+z_offset,radius)-V3(0,0,z_offset);
        V3 x10=Position(a1,z0+z_offset,radius)-V3(0,0,z_offset);
        V3 x01=Position(a0,z1+z_offset,radius)-V3(0,0,z_offset);
        V3 x11=Position(a1,z1+z_offset,radius)-V3(0,0,z_offset);
        V3 c00=Color(a0,z0+z_offset);
        V3 c10=Color(a1,z0+z_offset);
        V3 c01=Color(a0,z1+z_offset);
        V3 c11=Color(a1,z1+z_offset);
        renderer->DrawTriangle(x00,x10,x11,c00,c10,c11);
        renderer->DrawTriangle(x11,x01,x00,c11,c01,c00);
      }

      // for(int i=0;i<world.particles.size();i++)
      //   renderer->DrawPoint(world.particles[i].x,std::max(world.particles[i].r,0.01f)*10,world.particles[i].color);

      //Draw axis
    // renderer->DrawLine(V3(0,0,0),V3(1,0,0),0.01,V3(1,0,0),V3(1,0,0));
    // renderer->DrawLine(V3(0,0,0),V3(0,1,0),0.01,V3(0,1,0),V3(0,1,0));
    // renderer->DrawLine(V3(0,0,0),V3(0,0,1),0.01,V3(0,0,1),V3(0,0,1));

    renderer->SwapBuffers();
  }

  return 0;
}
